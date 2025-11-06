#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import matplotlib.pyplot as plt

from crypto.training import TrainingConfig, Trainer
from crypto.utils import plot_hedge_comparison


def main():

    # ========== Configuration ==========
    print("=" * 70)
    print("BITCOIN DEEP HEDGING - Training Framework Example")
    print("=" * 70)

    # Create training configuration
    config = TrainingConfig(
        strike=50000,
        maturity_days=14,
        call=True,
        volatility=0.8,
        drift=0.0,
        transaction_cost=0.0005,  # 0.05% (Deribit taker fee)
        dt_hours=8.0,  # 8-hour time steps (matches funding interval)
        n_paths=10000,
        n_epochs=80,
        n_layers=4,
        n_units=128,
        risk_measure="expected_shortfall",
        risk_param=0.9,
        model_path="../../models/deep_hedger_trained.pth",
        test_n_paths=200,
        test_seed=888,
        train_seed=42,
        device="cpu",  # Use "cuda" for GPU training
    )

    print(f"\n{config}")

    # Validate configuration
    config.validate()

    # ========== Train Model ==========

    # Create trainer
    trainer = Trainer(config)

    # Run full training pipeline
    results = trainer.train(seed=42)

    # Get training summary
    summary = results.summary()
    print(f"\nTraining Summary:")
    print(f"  Epochs: {summary['n_epochs']}")
    print(f"  Final loss: {summary['final_loss']:.6f}")
    print(f"  Improvement: {summary['improvement_pct']:.1f}%")

    # ========== Visualization (Optional) ==========

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS (using test option)")
    print("=" * 70)

    # Get model and test option from trainer for visualization
    model = trainer.model
    test_option = trainer.test_option

    print(f"\nGenerating hedge positions for visualization...")

    with torch.no_grad():
        # Compute deep hedging strategy on test set
        deep_hedge_positions = model.compute_hedge(test_option).squeeze(1)
        deep_hedge_pnl = model.compute_cum_pl(test_option)

        # Get spot prices
        spots = test_option.underlier.spot

        # Add funding costs if applicable
        if hasattr(test_option.underlier, "funding_rate") and hasattr(
            test_option.underlier, "funding_payment_times"
        ):
            from crypto.strategies.deep_hedge_utils import compute_funding_cum_cost

            funding_rate = test_option.underlier.funding_rate
            funding_times = test_option.underlier.funding_payment_times()

            deep_funding = compute_funding_cum_cost(
                spots=spots,
                positions=deep_hedge_positions,
                funding_rate=funding_rate,
                funding_times=funding_times,
            )
            deep_hedge_pnl = deep_hedge_pnl - deep_funding

        # Compute Black-Scholes baseline
        from crypto.strategies.deep_hedge_utils import calculate_bs_hedge_pnl

        bs_delta = test_option.black_scholes_delta()
        payoffs = test_option.payoff()
        cost = test_option.underlier.cost

        funding_rate = None
        funding_times = None
        if hasattr(test_option.underlier, "funding_rate"):
            funding_rate = test_option.underlier.funding_rate
            funding_times = test_option.underlier.funding_payment_times()

        bs_hedge_pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=cost,
            funding_rate=funding_rate,
            funding_times=funding_times,
        )

    print(f"✅ Generated positions for {spots.shape[0]} test paths")

    # ========== Performance Comparison ==========

    from crypto.strategies.deep_hedge_utils import compare_hedge_performance

    performance_results = compare_hedge_performance(deep_hedge_pnl, bs_hedge_pnl)

    from crypto.strategies.deep_hedge_utils import print_performance_comparison

    print_performance_comparison(performance_results)

    # ========== Plot Results ==========

    print("\n" + "=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    # Use utility function for comprehensive hedge comparison visualization
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"bitcoin_hedge_{config.n_epochs}epochs.png")

    # Get training history from results
    training_history = results.training_history

    fig = plot_hedge_comparison(
        deep_hedge_positions=deep_hedge_positions,
        bs_delta=bs_delta,
        deep_hedge_pnl=deep_hedge_pnl,
        bs_hedge_pnl=bs_hedge_pnl,
        spots=spots,
        strike=config.strike,
        training_history=training_history,
        performance_results=performance_results,
        path_idx=0,
        save_path=output_file,
    )
    plt.close()  # Close instead of show for non-interactive mode
    print(f"\n✅ Saved figure to {output_file}")

    # ========== Export Results ==========

    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)

    # Export training results to JSON
    results_dir = os.path.join(os.path.dirname(__file__), config.output_dir)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "training_results.json")

    results.to_json(results_file, include_raw=True, indent=2)
    print(f"\n✅ Training results saved to: {results_file}")

    print("\n" + "=" * 70)
    print("✅ Bitcoin deep hedging example complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Model: {results.model_path}")
    print(f"  Results: {results_file}")
    print(f"  Plot: {output_file}")
    print("=" * 70 + "\n")

    return {
        "trainer": trainer,
        "results": results,
        "model": model,
        "test_option": test_option,
        "deep_hedge_positions": deep_hedge_positions,
        "deep_hedge_pnl": deep_hedge_pnl,
        "bs_delta": bs_delta,
        "bs_hedge_pnl": bs_hedge_pnl,
        "spots": spots,
    }


if __name__ == "__main__":
    results = main()
