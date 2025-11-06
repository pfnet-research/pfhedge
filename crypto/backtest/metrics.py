import torch
from torch import Tensor
from typing import Union


def calculate_sharpe_ratio(pnl: Tensor, risk_free_rate: float = 0.0) -> float:
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
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    mean_pnl = pnl.mean().item()

    # Calculate downside deviation (only negative deviations from target)
    downside_returns = torch.clamp(pnl - target, max=0.0)
    downside_dev = torch.sqrt(torch.mean(downside_returns**2)).item()

    if downside_dev == 0:
        return 0.0

    return (mean_pnl - risk_free_rate) / downside_dev


def calculate_max_drawdown(pnl: Tensor) -> float:
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
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    # VaR is the alpha-quantile of the PnL distribution
    var = torch.quantile(pnl, alpha).item()

    return var


def calculate_win_rate(pnl: Tensor) -> float:
    # If cumulative PnL, take final values
    if pnl.dim() == 2:
        pnl = pnl[:, -1]

    n_wins = (pnl > 0).sum().item()
    n_total = len(pnl)

    return n_wins / n_total if n_total > 0 else 0.0


def calculate_calmar_ratio(pnl: Tensor) -> float:
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


def format_metrics(metrics: dict, name: str = "Strategy") -> str:
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"{name} Performance Metrics")
    lines.append(f"{'='*50}")

    # Basic statistics
    lines.append(f"\n{'Basic Statistics':^50}")
    lines.append(f"{'-'*50}")
    lines.append(f"{'Mean PnL':<30} ${metrics['mean']:>15.2f}")
    lines.append(f"{'Std Dev':<30} ${metrics['std']:>15.2f}")
    lines.append(f"{'Median PnL':<30} ${metrics['median']:>15.2f}")
    lines.append(f"{'Min PnL':<30} ${metrics['min']:>15.2f}")
    lines.append(f"{'Max PnL':<30} ${metrics['max']:>15.2f}")

    # Risk-adjusted returns
    lines.append(f"\n{'Risk-Adjusted Returns':^50}")
    lines.append(f"{'-'*50}")
    lines.append(f"{'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>18.3f}")
    lines.append(f"{'Sortino Ratio':<30} {metrics['sortino_ratio']:>18.3f}")
    if "calmar_ratio" in metrics:
        lines.append(f"{'Calmar Ratio':<30} {metrics['calmar_ratio']:>18.3f}")

    # Risk metrics
    lines.append(f"\n{'Risk Metrics':^50}")
    lines.append(f"{'-'*50}")
    for key in metrics:
        if "cvar" in key:
            lines.append(f"{key.upper():<30} ${metrics[key]:>15.2f}")
        elif "var" in key:
            lines.append(f"{key.upper():<30} ${metrics[key]:>15.2f}")
    if "max_drawdown" in metrics:
        lines.append(f"{'Max Drawdown':<30} ${metrics['max_drawdown']:>15.2f}")

    # Performance metrics
    lines.append(f"\n{'Performance Metrics':^50}")
    lines.append(f"{'-'*50}")
    lines.append(f"{'Win Rate':<30} {metrics['win_rate']:>17.1%}")

    lines.append(f"\n{'='*50}\n")

    return "\n".join(lines)


def print_metrics(metrics: dict, name: str = "Strategy") -> None:
    print(format_metrics(metrics, name))
