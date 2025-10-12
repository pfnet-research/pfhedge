"""Utilities for deep hedging strategies.

This module provides reusable functions for creating deep hedgers,
calculating baseline PnL, and comparing performance.
"""

import torch
from typing import Dict, Tuple
from pfhedge.nn import Hedger, MultiLayerPerceptron, ExpectedShortfall


# Default features for deep hedging (following snowball_hedge.py pattern)
DEFAULT_FEATURES = [
    "log_moneyness",
    "expiry_time",
    "volatility",
    "prev_hedge",
]


def create_deep_hedger(
    n_layers: int = 3,
    n_units: int = 64,
    risk_measure: str = "expected_shortfall",
    risk_param: float = 0.5,
    features: list = None,
) -> Hedger:
    """Create a deep hedger with standard configuration.

    Args:
        n_layers: Number of hidden layers
        n_units: Number of units per layer
        risk_measure: Risk measure type ('expected_shortfall' only for now)
        risk_param: Risk parameter (p for ExpectedShortfall)
        features: List of feature names (defaults to log_moneyness, expiry_time, volatility, prev_hedge)

    Returns:
        Configured Hedger instance
    """
    if features is None:
        features = DEFAULT_FEATURES

    # Create model
    model = MultiLayerPerceptron(
        n_layers=n_layers,
        n_units=[n_units] * n_layers
    )

    # Create criterion (only ExpectedShortfall for now)
    if risk_measure == "expected_shortfall":
        criterion = ExpectedShortfall(p=risk_param)
    else:
        raise ValueError(f"Unsupported risk measure: {risk_measure}")

    # Create hedger
    return Hedger(
        model=model,
        inputs=features,
        criterion=criterion
    )


def calculate_bs_hedge_pnl(
    spots: torch.Tensor,
    bs_delta: torch.Tensor,
    payoffs: torch.Tensor,
    cost: float,
) -> torch.Tensor:
    """Calculate Black-Scholes hedge PnL with transaction costs.

    This implements the manual PnL calculation accounting for:
    - Position changes (rebalancing)
    - Transaction costs from rebalancing
    - Final payoff at maturity

    Args:
        spots: Spot prices, shape (n_paths, n_steps)
        bs_delta: Black-Scholes delta, shape (n_paths, n_steps)
        payoffs: Option payoffs, shape (n_paths,)
        cost: Transaction cost rate (e.g., 0.001 for 0.1%)

    Returns:
        Cumulative PnL tensor, shape (n_paths, n_steps)
    """
    # Calculate position changes
    delta_diff = torch.zeros_like(bs_delta)
    delta_diff[:, 0] = bs_delta[:, 0]  # Initial position
    delta_diff[:, 1:] = bs_delta[:, 1:] - bs_delta[:, :-1]  # Rebalancing

    # Transaction costs from rebalancing
    transaction_costs = cost * torch.abs(delta_diff * spots)

    # PnL from holding delta positions
    spot_diff = torch.zeros_like(spots)
    spot_diff[:, 1:] = spots[:, 1:] - spots[:, :-1]
    position_pnl = bs_delta * spot_diff

    # Cumulative PnL
    cumulative_position_pnl = torch.cumsum(position_pnl, dim=1)
    cumulative_costs = torch.cumsum(transaction_costs, dim=1)

    # Final PnL = cumulative gains - costs - payoff at maturity
    bs_hedge_pnl = cumulative_position_pnl - cumulative_costs
    bs_hedge_pnl[:, -1] -= payoffs

    return bs_hedge_pnl


def compare_hedge_performance(
    deep_pnl: torch.Tensor,
    bs_pnl: torch.Tensor,
    names: Tuple[str, str] = ("Deep Hedge", "Black-Scholes"),
) -> Dict[str, Dict[str, float]]:
    """Compare performance of two hedging strategies.

    Args:
        deep_pnl: Deep hedging PnL, shape (n_paths, n_steps)
        bs_pnl: Baseline PnL, shape (n_paths, n_steps)
        names: Tuple of (deep_name, baseline_name)

    Returns:
        Dictionary with performance metrics for each strategy
    """
    # Get final PnL (last time step)
    deep_final = deep_pnl[:, -1] if deep_pnl.dim() == 2 else deep_pnl
    bs_final = bs_pnl[:, -1] if bs_pnl.dim() == 2 else bs_pnl

    # Calculate metrics
    results = {
        names[0]: {
            "mean": deep_final.mean().item(),
            "std": deep_final.std().item(),
            "min": deep_final.min().item(),
            "max": deep_final.max().item(),
        },
        names[1]: {
            "mean": bs_final.mean().item(),
            "std": bs_final.std().item(),
            "min": bs_final.min().item(),
            "max": bs_final.max().item(),
        }
    }

    # Add Sharpe ratios
    for name, pnl in [(names[0], deep_final), (names[1], bs_final)]:
        mean = results[name]["mean"]
        std = results[name]["std"]
        results[name]["sharpe"] = mean / std if std > 0 else 0.0

    return results


def print_performance_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty print performance comparison results.

    Args:
        results: Dictionary from compare_hedge_performance()
    """
    names = list(results.keys())

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<20} {names[0]:>15} {names[1]:>15}")
    print("-" * 52)

    print(f"{'Mean PnL':<20} ${results[names[0]]['mean']:>14.2f} ${results[names[1]]['mean']:>14.2f}")
    print(f"{'PnL Std':<20} ${results[names[0]]['std']:>14.2f} ${results[names[1]]['std']:>14.2f}")
    print(f"{'Sharpe Ratio':<20} {results[names[0]]['sharpe']:>15.3f} {results[names[1]]['sharpe']:>15.3f}")
    print(f"{'Min PnL':<20} ${results[names[0]]['min']:>14.2f} ${results[names[1]]['min']:>14.2f}")
    print(f"{'Max PnL':<20} ${results[names[0]]['max']:>14.2f} ${results[names[1]]['max']:>14.2f}")

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")

    deep_std = results[names[0]]['std']
    bs_std = results[names[1]]['std']

    if deep_std < bs_std:
        print(f"✅ {names[0]} achieves {(1 - deep_std/bs_std)*100:.1f}% lower risk")
    else:
        print(f"⚠️  {names[1]} has {(1 - bs_std/deep_std)*100:.1f}% lower risk")

    deep_sharpe = results[names[0]]['sharpe']
    bs_sharpe = results[names[1]]['sharpe']

    if deep_sharpe > bs_sharpe and bs_sharpe != 0:
        print(f"✅ {names[0]} has {((deep_sharpe/bs_sharpe - 1)*100):.1f}% better Sharpe ratio")
    elif bs_sharpe > deep_sharpe and deep_sharpe != 0:
        print(f"⚠️  {names[1]} has {((bs_sharpe/deep_sharpe - 1)*100):.1f}% better Sharpe ratio")
    elif bs_sharpe > deep_sharpe:
        print(f"⚠️  {names[1]} has better Sharpe ratio")

    deep_mean = results[names[0]]['mean']
    bs_mean = results[names[1]]['mean']

    if deep_mean > bs_mean:
        print(f"✅ {names[0]} has ${deep_mean - bs_mean:.2f} higher mean PnL")
    else:
        print(f"⚠️  {names[1]} has ${bs_mean - deep_mean:.2f} higher mean PnL")

    print(f"\n{'='*60}")
