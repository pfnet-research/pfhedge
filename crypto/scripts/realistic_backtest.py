#!/usr/bin/env python3

import argparse
import json
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, Optional

# Add parent directory to path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.training import TrainingConfig, Trainer
from crypto.backtest import BacktestConfig, Backtester
from crypto.utils.black_scholes import implied_volatility_from_btc_premium
from crypto.data.deribit_client import DeribitClient

# Import functions from package modules
from crypto.scripts.fetch_deribit_data import (
    fetch_perpetual_trades,
    resample_trades_to_ohlc,
    fetch_funding_rates,
    save_data,
)
from crypto.scripts.select_option import select_best_option

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_data_available(config: Dict, client: DeribitClient) -> bool:
    data_dir = Path(config.get("data_dir", "crypto/data/historical"))

    # Parse dates
    trade_date = datetime.fromisoformat(config["trade_date"]).replace(
        tzinfo=timezone.utc
    )
    expiry_date = datetime.fromisoformat(config["expiry_date"]).replace(
        tzinfo=timezone.utc
    )
    lookback_start = datetime.fromisoformat(
        config.get("lookback_start", config["trade_date"])
    ).replace(tzinfo=timezone.utc)

    # Add buffer for data fetching
    start_date = lookback_start - timedelta(days=1)
    end_date = expiry_date + timedelta(days=1)

    # Check if data already exists
    perpetual_file = (
        data_dir / f"btc-perpetual_8H_{start_date.date()}_{end_date.date()}.parquet"
    )
    funding_file = (
        data_dir
        / f"btc-perpetual_funding_{start_date.date()}_{end_date.date()}.parquet"
    )

    if perpetual_file.exists() and funding_file.exists():
        logger.info("Historical data already available")
        return True

    logger.info(f"Fetching historical data from {start_date} to {end_date}")

    # Fetch perpetual trades
    trades_df = fetch_perpetual_trades(
        client, start_date, end_date, instrument="BTC-PERPETUAL"
    )

    if trades_df.empty:
        logger.error("Failed to fetch perpetual trades")
        return False

    # Resample to 8H OHLC
    ohlc_df = resample_trades_to_ohlc(trades_df, frequency="8H")

    # Save perpetual data
    save_data(
        ohlc_df, data_dir, f"btc-perpetual_8H_{start_date.date()}_{end_date.date()}"
    )

    # Fetch funding rates
    funding_df = fetch_funding_rates(client, start_date, end_date)
    if not funding_df.empty:
        save_data(
            funding_df,
            data_dir,
            f"btc-perpetual_funding_{start_date.date()}_{end_date.date()}",
        )

    return True


def select_and_fetch_option(config: Dict, client: DeribitClient) -> Optional[Dict]:
    trade_date = datetime.fromisoformat(config["trade_date"]).replace(
        tzinfo=timezone.utc
    )
    expiry_date = datetime.fromisoformat(config["expiry_date"]).replace(
        tzinfo=timezone.utc
    )

    # Select best option
    option_info = select_best_option(
        client,
        trade_date,
        expiry_date,
        option_type=config.get("option_type", "call"),
        target_moneyness=config.get("moneyness_target", 1.0),
        min_trades=config.get("min_trade_count", 10),
    )

    if not option_info:
        logger.error("Failed to select option")
        return None

    # Save option metadata
    metadata_path = (
        Path(config.get("output_dir", "backtest_results")) / "option_metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(option_info, f, indent=2, default=str)

    logger.info(f"Selected option: {option_info['instrument_name']}")
    logger.info(
        f"Premium: {option_info['premium_btc']:.4f} BTC (${option_info['premium_usd']:,.2f})"
    )

    return option_info


def train_model_for_option(option_info: Dict, config: Dict) -> str:
    logger.info("Training deep hedging model with market parameters")

    # Calculate implied volatility from market premium
    time_to_expiry = option_info["days_to_expiry"] / 365.0

    # Check for volatility override in config
    if "volatility_override" in config and config["volatility_override"] is not None:
        iv = config["volatility_override"]
        logger.info(f"Using volatility override: {iv:.1%}")
    else:
        iv = implied_volatility_from_btc_premium(
            premium_btc=option_info["premium_btc"],
            spot=option_info["initial_spot"],
            strike=option_info["strike"],
            time_to_expiry=time_to_expiry,
            option_type=option_info["option_type"],
        )

        if iv is None:
            logger.warning("Could not calculate IV from premium, using default")
            iv = config.get("volatility_default", 0.8)  # Default 80% volatility

        logger.info(f"Implied volatility: {iv:.1%}")

    # Create training configuration
    training_config = config.get("training", {})

    train_cfg = TrainingConfig(
        strike=option_info["strike"],
        maturity_days=option_info["days_to_expiry"],
        call=(option_info["option_type"] == "call"),
        volatility=iv,
        transaction_cost=config.get("transaction_cost", 0.0006),
        dt_hours=config.get("dt_hours", 8.0),
        n_paths=training_config.get("n_paths", 50000),
        n_epochs=training_config.get("n_epochs", 100),
        n_layers=training_config.get("n_layers", 4),
        n_units=training_config.get("n_units", 128),
        risk_measure=training_config.get("risk_measure", "expected_shortfall"),
        risk_param=training_config.get("risk_param", 0.9),
    )

    # Set model path
    model_path = (
        Path(config.get("output_dir", "backtest_results")) / "trained_model.pth"
    )
    train_cfg.model_path = str(model_path)

    # Train model
    trainer = Trainer(train_cfg, verbose=True)
    results = trainer.train(seed=config.get("seed", 42))

    # Save training results
    results_path = (
        Path(config.get("output_dir", "backtest_results")) / "training_results.json"
    )
    results.to_json(str(results_path))

    logger.info(f"Model trained and saved to {model_path}")

    return str(model_path)


def run_backtest_with_model(option_info: Dict, model_path: str, config: Dict) -> Dict:
    logger.info("Running backtest from sale to expiry")

    # Parse dates
    trade_date = datetime.fromisoformat(option_info["trade_date"])
    expiry_date = datetime.fromisoformat(option_info["expiry_date"])

    # Create backtest configuration
    backtest_config = config.get("backtest", {})

    backtest_cfg = BacktestConfig(
        start_date=trade_date.strftime("%Y-%m-%d"),
        end_date=expiry_date.strftime("%Y-%m-%d"),
        strike=option_info["strike"],
        maturity_days=option_info["days_to_expiry"],
        model_path=model_path,
        call=(option_info["option_type"] == "call"),
        n_bootstrap_paths=backtest_config.get("n_bootstrap_paths", 1000),
        transaction_cost=config.get("transaction_cost", 0.0006),
        dt_hours=config.get("dt_hours", 8.0),
        data_dir=config.get("data_dir", "crypto/data/historical"),
        output_dir=config.get("output_dir", "backtest_results"),
    )

    # Run backtest
    backtester = Backtester(backtest_cfg)
    results = backtester.run(seed=config.get("seed", 42))

    return results


def calculate_seller_pnl(
    backtest_results, option_premium_btc: float, initial_spot: float
) -> Dict:
    logger.info("Calculating seller P&L including premium")

    # Convert premium to USD
    premium_usd = option_premium_btc * initial_spot

    # Get final hedging P&L
    deep_hedge_pnl = backtest_results.deep_pnl[:, -1].cpu().numpy()
    bs_hedge_pnl = backtest_results.bs_pnl[:, -1].cpu().numpy()

    # For seller: total P&L = premium received + hedging P&L
    # (hedging P&L is typically negative as we're hedging a short position)
    deep_total_pnl = premium_usd + deep_hedge_pnl
    bs_total_pnl = premium_usd + bs_hedge_pnl

    # Calculate metrics
    results = {
        "premium": {"btc": option_premium_btc, "usd": premium_usd},
        "deep_hedge": {
            "hedging_pnl": float(deep_hedge_pnl.mean()),
            "hedging_std": float(deep_hedge_pnl.std()),
            "total_pnl": float(deep_total_pnl.mean()),
            "total_std": float(deep_total_pnl.std()),
            "sharpe": (
                float(deep_total_pnl.mean() / deep_total_pnl.std())
                if deep_total_pnl.std() > 0
                else 0
            ),
            "win_rate": float((deep_total_pnl > 0).mean()),
            "max_loss": float(deep_total_pnl.min()),
            "max_profit": float(deep_total_pnl.max()),
        },
        "bs_hedge": {
            "hedging_pnl": float(bs_hedge_pnl.mean()),
            "hedging_std": float(bs_hedge_pnl.std()),
            "total_pnl": float(bs_total_pnl.mean()),
            "total_std": float(bs_total_pnl.std()),
            "sharpe": (
                float(bs_total_pnl.mean() / bs_total_pnl.std())
                if bs_total_pnl.std() > 0
                else 0
            ),
            "win_rate": float((bs_total_pnl > 0).mean()),
            "max_loss": float(bs_total_pnl.min()),
            "max_profit": float(bs_total_pnl.max()),
        },
        "comparison": {
            "pnl_improvement": float(deep_total_pnl.mean() - bs_total_pnl.mean()),
            "risk_reduction": (
                float((bs_total_pnl.std() - deep_total_pnl.std()) / bs_total_pnl.std())
                if bs_total_pnl.std() > 0
                else 0
            ),
            "sharpe_improvement": (
                float(
                    (deep_total_pnl.mean() / deep_total_pnl.std())
                    - (bs_total_pnl.mean() / bs_total_pnl.std())
                )
                if deep_total_pnl.std() > 0 and bs_total_pnl.std() > 0
                else 0
            ),
        },
    }

    return results


def generate_final_report(option_info: Dict, seller_pnl: Dict, config: Dict):
    output_dir = Path(config.get("output_dir", "backtest_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "realistic_backtest_report.md"

    with open(report_path, "w") as f:
        f.write("# Realistic Backtest Report\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")

        f.write("## Option Details\n\n")
        f.write(f"- **Instrument**: {option_info['instrument_name']}\n")
        f.write(f"- **Strike**: ${option_info['strike']:,.0f}\n")
        f.write(f"- **Type**: {option_info['option_type'].capitalize()}\n")
        f.write(f"- **Trade Date**: {option_info['trade_date']}\n")
        f.write(f"- **Expiry Date**: {option_info['expiry_date']}\n")
        f.write(f"- **Days to Expiry**: {option_info['days_to_expiry']}\n")
        f.write(f"- **Initial Spot**: ${option_info['initial_spot']:,.2f}\n")
        f.write(f"- **Moneyness**: {option_info['moneyness']:.3f}\n\n")

        f.write("## Premium Information\n\n")
        f.write(f"- **Premium (BTC)**: {seller_pnl['premium']['btc']:.4f}\n")
        f.write(f"- **Premium (USD)**: ${seller_pnl['premium']['usd']:,.2f}\n")
        f.write(f"- **Trade Count**: {option_info['trade_count']}\n\n")

        f.write("## Seller P&L Results\n\n")
        f.write("### Deep Hedge Strategy\n\n")
        dh = seller_pnl["deep_hedge"]
        f.write(
            f"- **Hedging P&L**: ${dh['hedging_pnl']:,.2f} ± ${dh['hedging_std']:,.2f}\n"
        )
        f.write(f"- **Premium Received**: ${seller_pnl['premium']['usd']:,.2f}\n")
        f.write(f"- **Total P&L**: ${dh['total_pnl']:,.2f} ± ${dh['total_std']:,.2f}\n")
        f.write(f"- **Sharpe Ratio**: {dh['sharpe']:.3f}\n")
        f.write(f"- **Win Rate**: {dh['win_rate']:.1%}\n")
        f.write(f"- **Max Loss**: ${dh['max_loss']:,.2f}\n")
        f.write(f"- **Max Profit**: ${dh['max_profit']:,.2f}\n\n")

        f.write("### Black-Scholes Baseline\n\n")
        bs = seller_pnl["bs_hedge"]
        f.write(
            f"- **Hedging P&L**: ${bs['hedging_pnl']:,.2f} ± ${bs['hedging_std']:,.2f}\n"
        )
        f.write(f"- **Premium Received**: ${seller_pnl['premium']['usd']:,.2f}\n")
        f.write(f"- **Total P&L**: ${bs['total_pnl']:,.2f} ± ${bs['total_std']:,.2f}\n")
        f.write(f"- **Sharpe Ratio**: {bs['sharpe']:.3f}\n")
        f.write(f"- **Win Rate**: {bs['win_rate']:.1%}\n")
        f.write(f"- **Max Loss**: ${bs['max_loss']:,.2f}\n")
        f.write(f"- **Max Profit**: ${bs['max_profit']:,.2f}\n\n")

        f.write("## Performance Comparison\n\n")
        comp = seller_pnl["comparison"]
        f.write(f"- **P&L Improvement**: ${comp['pnl_improvement']:,.2f}\n")
        f.write(f"- **Risk Reduction**: {comp['risk_reduction']:.1%}\n")
        f.write(f"- **Sharpe Improvement**: {comp['sharpe_improvement']:.3f}\n\n")

        f.write("## Configuration\n\n")
        f.write("```yaml\n")
        f.write(yaml.dump(config, default_flow_style=False))
        f.write("```\n")

    logger.info(f"Report saved to {report_path}")

    # Also save raw results as JSON
    results_path = output_dir / "seller_pnl_results.json"
    with open(results_path, "w") as f:
        json.dump(seller_pnl, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run realistic backtest")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    logger.info("Starting realistic backtest")
    logger.info(f"Configuration: {args.config}")

    # Initialize Deribit client
    testnet = config.get("testnet", False)
    client = DeribitClient(testnet=testnet)

    # 1. Ensure data is available
    if not ensure_data_available(config, client):
        logger.error("Failed to ensure data availability")
        sys.exit(1)

    # 2. Select option and fetch premium
    option_info = select_and_fetch_option(config, client)
    if not option_info:
        logger.error("Failed to select option")
        sys.exit(1)

    # 3. Train model with market parameters
    model_path = train_model_for_option(option_info, config)

    # 4. Run backtest
    backtest_results = run_backtest_with_model(option_info, model_path, config)

    # 5. Calculate seller P&L (including premium!)
    seller_pnl = calculate_seller_pnl(
        backtest_results, option_info["premium_btc"], option_info["initial_spot"]
    )

    # 6. Generate report
    generate_final_report(option_info, seller_pnl, config)

    # Print summary
    print("\n" + "=" * 60)
    print("REALISTIC BACKTEST COMPLETE")
    print("=" * 60)
    print(f"\nOption: {option_info['instrument_name']}")
    print(
        f"Premium received: {option_info['premium_btc']:.4f} BTC (${seller_pnl['premium']['usd']:,.2f})"
    )
    print(f"\nSeller P&L (including premium):")
    print(
        f"  Deep Hedge: ${seller_pnl['deep_hedge']['total_pnl']:,.2f} (Sharpe: {seller_pnl['deep_hedge']['sharpe']:.3f})"
    )
    print(
        f"  Black-Scholes: ${seller_pnl['bs_hedge']['total_pnl']:,.2f} (Sharpe: {seller_pnl['bs_hedge']['sharpe']:.3f})"
    )
    print(f"\nImprovement: ${seller_pnl['comparison']['pnl_improvement']:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
