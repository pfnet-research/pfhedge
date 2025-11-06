#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
import logging
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_option_metadata(option_file: str, instrument_name: str = None) -> dict:
    with open(option_file, "r") as f:
        data = json.load(f)

    # If data is a single option dict (from train_for_option.py output)
    if isinstance(data, dict) and "instrument_name" in data:
        if instrument_name and data["instrument_name"] != instrument_name:
            raise ValueError(
                f"Option file contains {data['instrument_name']}, "
                f"but requested {instrument_name}"
            )
        return data

    # If data is a list or wrapped in 'options' key (from explore_options.py)
    options = data if isinstance(data, list) else data.get("options", [])

    if not instrument_name:
        if len(options) == 1:
            return options[0]
        else:
            available = [opt["instrument_name"] for opt in options]
            raise ValueError(
                f"Option file contains multiple instruments. "
                f"Please specify --instrument:\n"
                + "\n".join(f"  - {name}" for name in available)
            )

    # Find matching instrument
    for opt in options:
        if opt["instrument_name"] == instrument_name:
            return opt

    available = [opt["instrument_name"] for opt in options]
    raise ValueError(
        f"Instrument '{instrument_name}' not found in {option_file}\n"
        f"Available:\n" + "\n".join(f"  - {name}" for name in available)
    )


def load_backtest_results(backtest_file: str) -> dict:
    with open(backtest_file, "r") as f:
        return json.load(f)


def calculate_seller_pnl(option: dict, backtest_results: dict) -> dict:
    # Extract premium
    premium_btc = option["premium_btc"]
    initial_spot = option["initial_spot"]
    premium_usd = premium_btc * initial_spot

    # Extract hedging P&L from backtest results
    # Results contain final PnL statistics for both strategies
    # Handle both legacy format (root level) and new format (summary level)
    if "summary" in backtest_results:
        deep_hedge = backtest_results["summary"]["deep_hedge"]
        bs_baseline = backtest_results["summary"]["bs_baseline"]
    else:
        deep_hedge = backtest_results["deep_hedge"]
        bs_baseline = backtest_results["bs_baseline"]

    # Get mean and std of hedging P&L
    deep_hedge_pnl_mean = deep_hedge["mean"]
    deep_hedge_pnl_std = deep_hedge["std"]

    bs_hedge_pnl_mean = bs_baseline["mean"]
    bs_hedge_pnl_std = bs_baseline["std"]

    # Calculate total seller P&L = Premium + Hedging P&L
    deep_total_pnl_mean = premium_usd + deep_hedge_pnl_mean
    deep_total_pnl_std = deep_hedge_pnl_std  # Std doesn't change (premium is constant)

    bs_total_pnl_mean = premium_usd + bs_hedge_pnl_mean
    bs_total_pnl_std = bs_hedge_pnl_std

    # Calculate Sharpe ratios
    deep_sharpe = (
        deep_total_pnl_mean / deep_total_pnl_std if deep_total_pnl_std > 0 else 0
    )
    bs_sharpe = bs_total_pnl_mean / bs_total_pnl_std if bs_total_pnl_std > 0 else 0

    # Build results
    results = {
        "option": {
            "instrument_name": option["instrument_name"],
            "strike": option["strike"],
            "option_type": option["option_type"],
            "initial_spot": initial_spot,
            "moneyness": option["moneyness"],
            "days_to_expiry": option["days_to_expiry"],
            "trade_date": option["trade_date"],
            "expiry_date": option["expiry_date"],
        },
        "premium": {"btc": premium_btc, "usd": premium_usd},
        "deep_hedge": {
            "hedging_pnl_mean": deep_hedge_pnl_mean,
            "hedging_pnl_std": deep_hedge_pnl_std,
            "total_pnl_mean": deep_total_pnl_mean,
            "total_pnl_std": deep_total_pnl_std,
            "sharpe_ratio": deep_sharpe,
            "win_rate": deep_hedge.get("win_rate", None),
            "max_drawdown": deep_hedge.get("max_drawdown", None),
            "cvar_95": deep_hedge.get("cvar_95", None),
        },
        "bs_baseline": {
            "hedging_pnl_mean": bs_hedge_pnl_mean,
            "hedging_pnl_std": bs_hedge_pnl_std,
            "total_pnl_mean": bs_total_pnl_mean,
            "total_pnl_std": bs_total_pnl_std,
            "sharpe_ratio": bs_sharpe,
            "win_rate": bs_baseline.get("win_rate", None),
            "max_drawdown": bs_baseline.get("max_drawdown", None),
            "cvar_95": bs_baseline.get("cvar_95", None),
        },
        "comparison": {
            "pnl_improvement": deep_total_pnl_mean - bs_total_pnl_mean,
            "risk_reduction_pct": (
                (bs_total_pnl_std - deep_total_pnl_std) / bs_total_pnl_std * 100
                if bs_total_pnl_std > 0
                else 0
            ),
            "sharpe_improvement": deep_sharpe - bs_sharpe,
        },
    }

    return results


def print_seller_analysis(results: dict):
    option = results["option"]
    premium = results["premium"]
    deep = results["deep_hedge"]
    bs = results["bs_baseline"]
    comp = results["comparison"]

    print("\n" + "=" * 100)
    print("SELLER P&L ANALYSIS")
    print("=" * 100)

    # Option details
    print(f"\nOption: {option['instrument_name']}")
    print(f"  Type: {option['option_type'].upper()}")
    print(f"  Strike: ${option['strike']:,.2f}")
    print(f"  Initial spot: ${option['initial_spot']:,.2f}")
    print(f"  Moneyness: {option['moneyness']:.3f}")
    print(f"  Days to expiry: {option['days_to_expiry']}")
    print(f"  Trade date: {option['trade_date']}")
    print(f"  Expiry date: {option['expiry_date']}")

    # Premium
    print(f"\nPremium Received (Income):")
    print(f"  {premium['btc']:.4f} BTC")
    print(f"  ${premium['usd']:,.2f} USD")

    # Deep hedge results
    print(f"\nDeep Hedge Strategy:")
    print(
        f"  Hedging P&L: ${deep['hedging_pnl_mean']:,.2f} ± ${deep['hedging_pnl_std']:,.2f}"
    )
    print(f"  Premium: ${premium['usd']:,.2f}")
    print(f"  {'─' * 50}")
    print(
        f"  Total Seller P&L: ${deep['total_pnl_mean']:,.2f} ± ${deep['total_pnl_std']:,.2f}"
    )
    print(f"  Sharpe Ratio: {deep['sharpe_ratio']:.3f}")
    if deep["win_rate"] is not None:
        print(f"  Win Rate: {deep['win_rate']:.1%}")
    if deep["max_drawdown"] is not None:
        print(f"  Max Drawdown: ${deep['max_drawdown']:,.2f}")
    if deep["cvar_95"] is not None:
        print(f"  CVaR (95%): ${deep['cvar_95']:,.2f}")

    # BS baseline results
    print(f"\nBlack-Scholes Baseline:")
    print(
        f"  Hedging P&L: ${bs['hedging_pnl_mean']:,.2f} ± ${bs['hedging_pnl_std']:,.2f}"
    )
    print(f"  Premium: ${premium['usd']:,.2f}")
    print(f"  {'─' * 50}")
    print(
        f"  Total Seller P&L: ${bs['total_pnl_mean']:,.2f} ± ${bs['total_pnl_std']:,.2f}"
    )
    print(f"  Sharpe Ratio: {bs['sharpe_ratio']:.3f}")
    if bs["win_rate"] is not None:
        print(f"  Win Rate: {bs['win_rate']:.1%}")
    if bs["max_drawdown"] is not None:
        print(f"  Max Drawdown: ${bs['max_drawdown']:,.2f}")
    if bs["cvar_95"] is not None:
        print(f"  CVaR (95%): ${bs['cvar_95']:,.2f}")

    # Comparison
    print(f"\nComparison (Deep Hedge vs Black-Scholes):")
    print(f"  P&L Improvement: ${comp['pnl_improvement']:+,.2f}")
    print(f"  Risk Reduction: {comp['risk_reduction_pct']:+.1f}%")
    print(f"  Sharpe Improvement: {comp['sharpe_improvement']:+.3f}")

    # Trading decision
    print(f"\n{'=' * 100}")
    print("TRADING DECISION:")
    print("=" * 100)

    if deep["total_pnl_mean"] > 0:
        print(
            f"✓ PROFITABLE: Expected to make ${deep['total_pnl_mean']:,.2f} per option"
        )
        print(
            f"  Premium ({premium['usd']:,.2f}) exceeds hedging costs ({abs(deep['hedging_pnl_mean']):,.2f})"
        )
        print(f"  Deep hedge Sharpe ratio: {deep['sharpe_ratio']:.3f}")
    else:
        print(
            f"✗ UNPROFITABLE: Expected to lose ${abs(deep['total_pnl_mean']):,.2f} per option"
        )
        print(
            f"  Hedging costs ({abs(deep['hedging_pnl_mean']):,.2f}) exceed premium ({premium['usd']:,.2f})"
        )
        print(f"  Consider: Higher premium, shorter expiry, or different strike")

    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate seller P&L from option premium and backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--option",
        required=True,
        help="Option metadata JSON file (from explore_options.py or train_for_option.py)",
    )
    parser.add_argument(
        "--backtest",
        required=True,
        help="Backtest results JSON file (from crypto.backtest.run)",
    )

    # Optional arguments
    parser.add_argument(
        "--instrument", help="Instrument name if option file contains multiple options"
    )
    parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    args = parser.parse_args()

    # Load option metadata
    try:
        option = load_option_metadata(args.option, args.instrument)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError loading option metadata: {e}\n")
        return 1

    # Load backtest results
    try:
        backtest_results = load_backtest_results(args.backtest)
    except FileNotFoundError:
        print(f"\nError: Backtest results file not found: {args.backtest}")
        print("Run backtest first using: python -m crypto.backtest.run\n")
        return 1
    except Exception as e:
        print(f"\nError loading backtest results: {e}\n")
        return 1

    # Calculate seller P&L
    try:
        results = calculate_seller_pnl(option, backtest_results)
    except Exception as e:
        print(f"\nError calculating seller P&L: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Print analysis
    print_seller_analysis(results)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Results saved to: {output_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
