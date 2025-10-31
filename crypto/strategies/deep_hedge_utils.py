"""Utilities for deep hedging strategies.

This module provides reusable functions for creating deep hedgers,
calculating baseline PnL, and comparing performance.
"""

import torch
from typing import Dict, Tuple, Optional
from pfhedge.nn import (
    Hedger,
    MultiLayerPerceptron,
    ExpectedShortfall,
    EntropicRiskMeasure,
    QuadraticCVaR,
)
from pfhedge.nn.modules.loss import EntropicLoss


# Default features for deep hedging
# Note: PFHedge expects "expiry_time" for European options
DEFAULT_FEATURES = [
    "log_moneyness",
    "expiry_time",  # Changed from time_to_maturity to match PFHedge expectations
    "volatility",
    "prev_hedge",
]


def create_deep_hedger(
    n_layers: int = 3,
    n_units: "int | list[int]" = 64,  # Can be int or list of ints
    risk_measure: str = "expected_shortfall",
    risk_param: float = 0.5,
    features: list = None,
) -> Hedger:
    """Create a deep hedger with standard configuration.

    Args:
        n_layers: Number of hidden layers
        n_units: Number of units per layer. Can be:
            - int: Same units for all layers (e.g., 64 → [64, 64, 64])
            - list of ints: Variable units per layer (e.g., [64, 32, 32, 16])
        risk_measure: Risk measure type
            - 'expected_shortfall': ExpectedShortfall (CVaR)
            - 'entropic': EntropicRiskMeasure (exponential utility risk measure)
            - 'entropic_loss': EntropicLoss (expected exponential utility)
            - 'quadratic_cvar': QuadraticCVaR (Buehler 2019)
        risk_param: Risk parameter
            - For ExpectedShortfall: p (quantile level, 0 < p <= 1)
            - For EntropicRiskMeasure/EntropicLoss: a (risk aversion, a > 0)
            - For QuadraticCVaR: lam (lambda, lam >= 1)
        features: List of feature names (defaults to log_moneyness, time_to_maturity, volatility, prev_hedge)

    Returns:
        Configured Hedger instance
    """
    if features is None:
        features = DEFAULT_FEATURES

    # Create model with variable or uniform layer sizes
    if isinstance(n_units, list):
        units_list = n_units
    else:
        units_list = [n_units] * n_layers

    model = MultiLayerPerceptron(n_layers=n_layers, n_units=units_list)

    # Create criterion based on risk measure
    if risk_measure == "expected_shortfall":
        criterion = ExpectedShortfall(p=risk_param)
    elif risk_measure == "entropic":
        criterion = EntropicRiskMeasure(a=risk_param)
    elif risk_measure == "entropic_loss":
        criterion = EntropicLoss(a=risk_param)
    elif risk_measure == "quadratic_cvar":
        criterion = QuadraticCVaR(lam=risk_param)
    else:
        raise ValueError(
            f"Unsupported risk measure: {risk_measure}. "
            f"Choose from: expected_shortfall, entropic, entropic_loss, quadratic_cvar"
        )

    # Create hedger
    return Hedger(model=model, inputs=features, criterion=criterion)


def calculate_bs_hedge_pnl(
    spots: torch.Tensor,
    bs_delta: torch.Tensor,
    payoffs: torch.Tensor,
    cost: float,
    funding_rate: Optional[torch.Tensor] = None,
    funding_times: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Calculate Black-Scholes hedge PnL with transaction costs.

    This implements PnL calculation following PFHedge's cum_pl logic:
    - Capital gains: δ_{i-1} * (S_i - S_{i-1}) using PREVIOUS position
    - Transaction costs: applied to new spot prices after trades
    - Final payoff subtracted at maturity

    Args:
        spots: Spot prices, shape (n_paths, n_steps)
        bs_delta: Black-Scholes delta, shape (n_paths, n_steps)
        payoffs: Option payoffs, shape (n_paths,)
        cost: Transaction cost rate (e.g., 0.001 for 0.1%)

    Returns:
        Cumulative PnL tensor, shape (n_paths, n_steps)
    """
    # Capital gains: δ_{i-1} * (S_i - S_{i-1})
    # Use PREVIOUS position (not current) for price changes
    capital_gains = torch.cat(
        [
            torch.zeros_like(spots[:, [0]]),  # No gain at first step
            bs_delta[:, :-1]
            * (spots[:, 1:] - spots[:, :-1]),  # Previous delta * price change
        ],
        dim=1,
    )

    # Cumulative capital gains
    cumulative_pnl = capital_gains.cumsum(dim=1)

    # Subtract payoff at maturity
    cumulative_pnl[:, -1] -= payoffs

    # Transaction costs
    if cost > 0:
        # Position changes: |δ_i - δ_{i-1}|
        delta_changes = torch.cat(
            [
                bs_delta[:, [0]],  # Initial position
                bs_delta[:, 1:] - bs_delta[:, :-1],  # Rebalancing
            ],
            dim=1,
        )

        # Transaction costs applied to spot prices AFTER trade
        # First cost uses initial spot, subsequent costs use new spots
        transaction_costs = cost * torch.abs(delta_changes * spots)

        # Cumulative transaction costs
        cumulative_costs = transaction_costs.cumsum(dim=1)

        # Subtract costs from PnL
        cumulative_pnl -= cumulative_costs

    # Subtract funding costs if provided
    if funding_rate is not None and funding_times is not None:
        cumulative_funding = compute_funding_cum_cost(
            spots=spots,
            positions=bs_delta,
            funding_rate=funding_rate,
            funding_times=funding_times,
        )
        cumulative_pnl = cumulative_pnl - cumulative_funding

    return cumulative_pnl


def compute_funding_cum_cost(
    spots: torch.Tensor,
    positions: torch.Tensor,
    funding_rate: torch.Tensor,
    funding_times: torch.Tensor,
) -> torch.Tensor:
    """Compute cumulative funding cost for positions.

    Args:
        spots: Spot prices, shape (n_paths, n_steps)
        positions: Hedge positions, shape (n_paths, n_steps)
        funding_rate: Funding rates, shape (n_paths, n_steps)
        funding_times: Boolean mask of payment times, shape (n_steps,) or (n_paths, n_steps)

    Returns:
        Cumulative funding cost (positive = cost), shape (n_paths, n_steps)
    """
    if funding_times.dim() == 1:
        funding_times = funding_times.unsqueeze(0).expand_as(spots)

    # Do not charge funding at t=0
    funding_times = funding_times.clone()
    funding_times[..., 0] = False

    payments = positions * funding_rate * spots
    payments = payments * funding_times.to(spots.dtype)

    cumulative_funding = payments.cumsum(dim=1)
    return cumulative_funding


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
        },
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

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<20} {names[0]:>15} {names[1]:>15}")
    print("-" * 52)

    print(
        f"{'Mean PnL':<20} ${results[names[0]]['mean']:>14.2f} ${results[names[1]]['mean']:>14.2f}"
    )
    print(
        f"{'PnL Std':<20} ${results[names[0]]['std']:>14.2f} ${results[names[1]]['std']:>14.2f}"
    )
    print(
        f"{'Sharpe Ratio':<20} {results[names[0]]['sharpe']:>15.3f} {results[names[1]]['sharpe']:>15.3f}"
    )
    print(
        f"{'Min PnL':<20} ${results[names[0]]['min']:>14.2f} ${results[names[1]]['min']:>14.2f}"
    )
    print(
        f"{'Max PnL':<20} ${results[names[0]]['max']:>14.2f} ${results[names[1]]['max']:>14.2f}"
    )

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")

    deep_std = results[names[0]]["std"]
    bs_std = results[names[1]]["std"]

    if deep_std < bs_std:
        print(f"✅ {names[0]} achieves {(1 - deep_std/bs_std)*100:.1f}% lower risk")
    else:
        print(f"⚠️  {names[1]} has {(1 - bs_std/deep_std)*100:.1f}% lower risk")

    deep_sharpe = results[names[0]]["sharpe"]
    bs_sharpe = results[names[1]]["sharpe"]

    if deep_sharpe > bs_sharpe and bs_sharpe != 0:
        print(
            f"✅ {names[0]} has {((deep_sharpe/bs_sharpe - 1)*100):.1f}% better Sharpe ratio"
        )
    elif bs_sharpe > deep_sharpe and deep_sharpe != 0:
        print(
            f"⚠️  {names[1]} has {((bs_sharpe/deep_sharpe - 1)*100):.1f}% better Sharpe ratio"
        )
    elif bs_sharpe > deep_sharpe:
        print(f"⚠️  {names[1]} has better Sharpe ratio")

    deep_mean = results[names[0]]["mean"]
    bs_mean = results[names[1]]["mean"]

    if deep_mean > bs_mean:
        print(f"✅ {names[0]} has ${deep_mean - bs_mean:.2f} higher mean PnL")
    else:
        print(f"⚠️  {names[1]} has ${bs_mean - deep_mean:.2f} higher mean PnL")

    print(f"\n{'='*60}")
