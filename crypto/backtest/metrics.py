"""Performance metrics for backtesting.

This module provides functions to calculate various risk and performance metrics
for evaluating hedging strategies.
"""

import torch
from torch import Tensor
from typing import Union


def calculate_sharpe_ratio(pnl: Tensor, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio.

    The Sharpe ratio measures risk-adjusted return as the ratio of mean excess
    return to standard deviation.

    Args:
        pnl: PnL tensor, shape (n_paths,) for final PnL or (n_paths, n_steps) for cumulative
        risk_free_rate: Risk-free rate (default: 0.0)

    Returns:
        Sharpe ratio (float). Returns 0.0 if standard deviation is zero.

    Examples:
        >>> pnl = torch.tensor([100.0, 150.0, 80.0, 120.0, 110.0])
        >>> sharpe = calculate_sharpe_ratio(pnl)
        >>> print(f"Sharpe: {sharpe:.3f}")
    """
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    mean_pnl = pnl.mean().item()
    std_pnl = pnl.std().item()

    if std_pnl == 0:
        return 0.0

    return (mean_pnl - risk_free_rate) / std_pnl


def calculate_sortino_ratio(
    pnl: Tensor, risk_free_rate: float = 0.0, target: float = 0.0
) -> float:
    """Calculate Sortino ratio.

    The Sortino ratio is similar to Sharpe but only penalizes downside volatility.
    It uses downside deviation instead of total standard deviation.

    Args:
        pnl: PnL tensor, shape (n_paths,) for final PnL or (n_paths, n_steps) for cumulative
        risk_free_rate: Risk-free rate (default: 0.0)
        target: Target return for downside calculation (default: 0.0)

    Returns:
        Sortino ratio (float). Returns 0.0 if downside deviation is zero.

    Examples:
        >>> pnl = torch.tensor([100.0, 150.0, -50.0, 120.0, -20.0])
        >>> sortino = calculate_sortino_ratio(pnl)
    """
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    mean_pnl = pnl.mean().item()

    # Calculate downside deviation (only negative deviations from target)
    downside_returns = torch.clamp(pnl - target, max=0.0)
    downside_dev = torch.sqrt(torch.mean(downside_returns ** 2)).item()

    if downside_dev == 0:
        return 0.0

    return (mean_pnl - risk_free_rate) / downside_dev


def calculate_max_drawdown(pnl: Tensor) -> float:
    """Calculate maximum drawdown.

    Maximum drawdown is the largest peak-to-trough decline in cumulative PnL.

    Args:
        pnl: Cumulative PnL tensor, shape (n_paths, n_steps) or (n_steps,)

    Returns:
        Maximum drawdown (positive value represents loss)

    Examples:
        >>> cum_pnl = torch.tensor([[0., 10., 15., 8., 12., 5.]])
        >>> max_dd = calculate_max_drawdown(cum_pnl)
        >>> print(f"Max Drawdown: ${max_dd:.2f}")
    """
    # Handle 1D case
    if pnl.dim() == 1:
        pnl = pnl.unsqueeze(0)

    # For each path, calculate max drawdown
    drawdowns = []
    for path_pnl in pnl:
        # Calculate running maximum
        running_max = torch.cummax(path_pnl, dim=0)[0]
        # Drawdown at each point
        dd = running_max - path_pnl
        # Max drawdown for this path
        max_dd = dd.max().item()
        drawdowns.append(max_dd)

    # Return average max drawdown across paths
    return sum(drawdowns) / len(drawdowns)


def calculate_cvar(pnl: Tensor, alpha: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    CVaR is the expected loss given that the loss exceeds the VaR threshold.
    It measures tail risk.

    Args:
        pnl: PnL tensor, shape (n_paths,) for final PnL or (n_paths, n_steps) for cumulative
        alpha: Confidence level (default: 0.05 for 95% CVaR)

    Returns:
        CVaR value (negative means loss)

    Examples:
        >>> pnl = torch.randn(1000) * 100  # Random PnL
        >>> cvar_95 = calculate_cvar(pnl, alpha=0.05)  # 95% CVaR
        >>> cvar_99 = calculate_cvar(pnl, alpha=0.01)  # 99% CVaR
    """
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    # Sort PnL in ascending order (worst losses first)
    sorted_pnl = torch.sort(pnl)[0]

    # Calculate number of samples in the tail
    n_tail = max(1, int(alpha * len(sorted_pnl)))

    # CVaR is the mean of the worst alpha% of outcomes
    cvar = sorted_pnl[:n_tail].mean().item()

    return cvar


def calculate_var(pnl: Tensor, alpha: float = 0.05) -> float:
    """Calculate Value at Risk (VaR).

    VaR is the loss threshold at a given confidence level.
    For example, 95% VaR is the loss exceeded in 5% of worst cases.

    Args:
        pnl: PnL tensor, shape (n_paths,) for final PnL or (n_paths, n_steps) for cumulative
        alpha: Confidence level (default: 0.05 for 95% VaR)

    Returns:
        VaR value (negative means loss)

    Examples:
        >>> pnl = torch.randn(1000) * 100
        >>> var_95 = calculate_var(pnl, alpha=0.05)
        >>> print(f"95% VaR: ${var_95:.2f}")
    """
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    # VaR is the alpha-quantile of the PnL distribution
    var = torch.quantile(pnl, alpha).item()

    return var


def calculate_win_rate(pnl: Tensor) -> float:
    """Calculate win rate (percentage of profitable outcomes).

    Args:
        pnl: PnL tensor, shape (n_paths,) for final PnL or (n_paths, n_steps) for cumulative

    Returns:
        Win rate as a fraction between 0 and 1

    Examples:
        >>> pnl = torch.tensor([100., -50., 30., 80., -20.])
        >>> win_rate = calculate_win_rate(pnl)
        >>> print(f"Win rate: {win_rate:.1%}")
    """
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    n_wins = (pnl > 0).sum().item()
    n_total = len(pnl)

    return n_wins / n_total if n_total > 0 else 0.0


def calculate_calmar_ratio(pnl: Tensor) -> float:
    """Calculate Calmar ratio (return / max drawdown).

    The Calmar ratio measures return relative to maximum drawdown.
    Higher is better. Requires cumulative PnL to calculate drawdown.

    Args:
        pnl: Cumulative PnL tensor, shape (n_paths, n_steps)

    Returns:
        Calmar ratio (float). Returns 0.0 if max drawdown is zero.

    Examples:
        >>> cum_pnl = torch.randn(100, 50).cumsum(dim=1)
        >>> calmar = calculate_calmar_ratio(cum_pnl)
    """
    if pnl.dim() == 1:
        raise ValueError("Calmar ratio requires cumulative PnL (2D tensor)")

    # Calculate mean final return
    final_pnl = pnl[:, -1]
    mean_return = final_pnl.mean().item()

    # Calculate max drawdown
    max_dd = calculate_max_drawdown(pnl)

    if max_dd == 0:
        return 0.0

    return mean_return / max_dd


def calculate_all_metrics(
    pnl: Tensor,
    cumulative_pnl: Union[Tensor, None] = None,
    alpha_cvar: float = 0.05,
    alpha_var: float = 0.05,
) -> dict:
    """Calculate all metrics at once.

    Args:
        pnl: Final PnL tensor, shape (n_paths,)
        cumulative_pnl: Cumulative PnL tensor, shape (n_paths, n_steps), optional
        alpha_cvar: Confidence level for CVaR (default: 0.05)
        alpha_var: Confidence level for VaR (default: 0.05)

    Returns:
        Dictionary with all calculated metrics

    Examples:
        >>> final_pnl = torch.randn(1000) * 100
        >>> cum_pnl = torch.randn(1000, 50).cumsum(dim=1)
        >>> metrics = calculate_all_metrics(final_pnl, cum_pnl)
        >>> print(metrics['sharpe_ratio'])
    """
    metrics = {
        # Basic statistics
        "mean": pnl.mean().item(),
        "std": pnl.std().item(),
        "min": pnl.min().item(),
        "max": pnl.max().item(),
        "median": pnl.median().item(),
        # Risk-adjusted returns
        "sharpe_ratio": calculate_sharpe_ratio(pnl),
        "sortino_ratio": calculate_sortino_ratio(pnl),
        # Risk metrics
        f"cvar_{int((1-alpha_cvar)*100)}": calculate_cvar(pnl, alpha_cvar),
        f"var_{int((1-alpha_var)*100)}": calculate_var(pnl, alpha_var),
        # Performance metrics
        "win_rate": calculate_win_rate(pnl),
    }

    # Add metrics that require cumulative PnL
    if cumulative_pnl is not None:
        metrics["max_drawdown"] = calculate_max_drawdown(cumulative_pnl)
        metrics["calmar_ratio"] = calculate_calmar_ratio(cumulative_pnl)

    return metrics


def print_metrics(metrics: dict, name: str = "Strategy") -> None:
    """Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metrics from calculate_all_metrics()
        name: Name of the strategy (for display)

    Examples:
        >>> metrics = calculate_all_metrics(pnl)
        >>> print_metrics(metrics, name="Deep Hedge")
    """
    print(f"\n{'='*50}")
    print(f"{name} Performance Metrics")
    print(f"{'='*50}")

    # Basic statistics
    print(f"\n{'Basic Statistics':^50}")
    print(f"{'-'*50}")
    print(f"{'Mean PnL':<30} ${metrics['mean']:>15.2f}")
    print(f"{'Std Dev':<30} ${metrics['std']:>15.2f}")
    print(f"{'Median PnL':<30} ${metrics['median']:>15.2f}")
    print(f"{'Min PnL':<30} ${metrics['min']:>15.2f}")
    print(f"{'Max PnL':<30} ${metrics['max']:>15.2f}")

    # Risk-adjusted returns
    print(f"\n{'Risk-Adjusted Returns':^50}")
    print(f"{'-'*50}")
    print(f"{'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>18.3f}")
    print(f"{'Sortino Ratio':<30} {metrics['sortino_ratio']:>18.3f}")
    if "calmar_ratio" in metrics:
        print(f"{'Calmar Ratio':<30} {metrics['calmar_ratio']:>18.3f}")

    # Risk metrics
    print(f"\n{'Risk Metrics':^50}")
    print(f"{'-'*50}")
    for key in metrics:
        if "cvar" in key:
            print(f"{key.upper():<30} ${metrics[key]:>15.2f}")
        elif "var" in key:
            print(f"{key.upper():<30} ${metrics[key]:>15.2f}")
    if "max_drawdown" in metrics:
        print(f"{'Max Drawdown':<30} ${metrics['max_drawdown']:>15.2f}")

    # Performance metrics
    print(f"\n{'Performance Metrics':^50}")
    print(f"{'-'*50}")
    print(f"{'Win Rate':<30} {metrics['win_rate']:>17.1%}")

    print(f"\n{'='*50}\n")
