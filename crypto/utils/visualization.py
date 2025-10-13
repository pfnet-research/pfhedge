"""
Visualization utilities for crypto deep hedging framework.

This module provides reusable plotting functions that work with any instrument
in our framework (BitcoinSpot, BitcoinPerpetualBrownian, BitcoinPerpetualHistorical, etc.)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List
from datetime import datetime, timedelta


def plot_price_paths(
    instrument,
    n_paths_to_show: int = 10,
    time_unit: str = "days",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_stats: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot price paths for any instrument in our framework.

    Args:
        instrument: Any instrument with .spot attribute (BitcoinSpot, BitcoinPerpetual*, etc.)
        n_paths_to_show: Number of price paths to display
        time_unit: Time unit for x-axis ('days', 'hours', 'minutes')
        title: Custom title for the plot
        figsize: Figure size (width, height)
        show_stats: Whether to show price statistics
        save_path: Path to save the plot (optional)

    Returns:
        matplotlib.Figure: The created figure

    Examples:
        >>> btc = BitcoinPerpetualBrownian(sigma=0.8)
        >>> btc.simulate(n_paths=1000, time_horizon=30/365)
        >>> fig = plot_price_paths(btc, n_paths_to_show=20, time_unit="days")
        >>> plt.show()

        >>> btc_hist = BitcoinPerpetualHistorical(data_loader=loader)
        >>> btc_hist.simulate(n_paths=1, time_horizon=7/365)
        >>> fig = plot_price_paths(btc_hist, title="Historical Bitcoin Prices")
    """

    if not hasattr(instrument, "spot"):
        raise ValueError("Instrument must have a 'spot' attribute with price data")

    spot_prices = instrument.spot
    if spot_prices.dim() != 2:
        raise ValueError("Spot prices must be 2D tensor (n_paths, n_steps)")

    n_paths, n_steps = spot_prices.shape
    n_paths_to_show = min(n_paths_to_show, n_paths)

    # Convert to numpy for plotting
    prices_np = spot_prices.detach().numpy()

    # Create time axis
    if hasattr(instrument, "dt"):
        dt = instrument.dt
        if time_unit == "days":
            time_axis = np.linspace(0, dt * (n_steps - 1) * 365, n_steps)
            time_label = "Days"
        elif time_unit == "hours":
            time_axis = np.linspace(0, dt * (n_steps - 1) * 365 * 24, n_steps)
            time_label = "Hours"
        elif time_unit == "minutes":
            time_axis = np.linspace(0, dt * (n_steps - 1) * 365 * 24 * 60, n_steps)
            time_label = "Minutes"
        else:
            time_axis = np.arange(n_steps)
            time_label = "Time Steps"
    else:
        time_axis = np.arange(n_steps)
        time_label = "Time Steps"

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Price paths
    ax1 = axes[0]

    # Plot sample paths
    alpha = min(0.8, 20 / n_paths_to_show)  # Adjust transparency
    for i in range(n_paths_to_show):
        ax1.plot(time_axis, prices_np[i], alpha=alpha, linewidth=1)

    # Add mean path if multiple paths
    if n_paths > 1:
        mean_path = prices_np.mean(axis=0)
        ax1.plot(
            time_axis, mean_path, "r-", linewidth=2, label=f"Mean ({n_paths} paths)"
        )
        ax1.legend()

    ax1.set_title(title or f"{instrument.__class__.__name__} Price Paths")
    ax1.set_xlabel(time_label)
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)

    # Right plot: Final price distribution
    ax2 = axes[1]
    final_prices = prices_np[:, -1]

    n_bins = min(30, max(10, n_paths // 10))  # Adaptive bins
    ax2.hist(final_prices, bins=n_bins, alpha=0.7, edgecolor="black", density=True)
    ax2.axvline(
        final_prices.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: ${final_prices.mean():.0f}",
    )

    if show_stats:
        # Add percentiles
        p5, p95 = np.percentile(final_prices, [5, 95])
        ax2.axvline(
            p5, color="orange", linestyle=":", alpha=0.7, label=f"5%: ${p5:.0f}"
        )
        ax2.axvline(
            p95, color="orange", linestyle=":", alpha=0.7, label=f"95%: ${p95:.0f}"
        )

    ax2.set_title("Final Price Distribution")
    ax2.set_xlabel("Final Price ($)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Print summary statistics
    if show_stats:
        print(f"\n📊 Price Path Statistics:")
        print(f"  Paths: {n_paths:,}")
        print(f"  Time steps: {n_steps}")
        print(f"  Initial price: ${prices_np[0, 0]:.2f}")
        print(f"  Final price (mean): ${final_prices.mean():.2f}")
        print(f"  Final price (std): ${final_prices.std():.2f}")
        print(f"  Price range: ${final_prices.min():.0f} - ${final_prices.max():.0f}")

        # Calculate returns statistics
        total_returns = (final_prices / prices_np[:, 0] - 1) * 100
        print(f"  Total return (mean): {total_returns.mean():.1f}%")
        print(f"  Total return (std): {total_returns.std():.1f}%")

    return fig


def plot_option_analysis(
    instrument,
    strike: float,
    option_type: str = "call",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Analyze option payoffs for any instrument.

    Args:
        instrument: Any instrument with .spot attribute
        strike: Strike price for the option
        option_type: 'call' or 'put'
        title: Custom title
        figsize: Figure size
        save_path: Path to save plot

    Returns:
        matplotlib.Figure: The created figure
    """

    if not hasattr(instrument, "spot"):
        raise ValueError("Instrument must have a 'spot' attribute")

    final_prices = instrument.spot[:, -1].detach().numpy()

    # Calculate payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(final_prices - strike, 0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(strike - final_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Payoff diagram
    ax1 = axes[0]
    scatter_alpha = min(1.0, 100 / len(final_prices))
    ax1.scatter(final_prices, payoffs, alpha=scatter_alpha)
    ax1.axvline(strike, color="red", linestyle="--", label=f"Strike: ${strike:.0f}")
    ax1.set_title(f"{option_type.title()} Option Payoff")
    ax1.set_xlabel("Final Price ($)")
    ax1.set_ylabel("Payoff ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Payoff distribution (ITM only)
    ax2 = axes[1]
    itm_payoffs = payoffs[payoffs > 0]
    if len(itm_payoffs) > 0:
        ax2.hist(itm_payoffs, bins=20, alpha=0.7, edgecolor="black")
        ax2.axvline(
            itm_payoffs.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: ${itm_payoffs.mean():.0f}",
        )
        ax2.set_title("ITM Payoff Distribution")
        ax2.set_xlabel("Payoff ($)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
    else:
        ax2.text(
            0.5,
            0.5,
            "No ITM Options",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=14,
        )
        ax2.set_title("ITM Payoff Distribution")
    ax2.grid(True, alpha=0.3)

    # Moneyness analysis
    ax3 = axes[2]
    moneyness = final_prices / strike
    ax3.hist(moneyness, bins=25, alpha=0.7, edgecolor="black")
    ax3.axvline(1.0, color="red", linestyle="--", label="ATM")
    ax3.axvline(
        moneyness.mean(),
        color="orange",
        linestyle="--",
        label=f"Mean: {moneyness.mean():.2f}",
    )
    ax3.set_title("Moneyness Distribution")
    ax3.set_xlabel("S/K Ratio")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Print option statistics
    itm_ratio = (payoffs > 0).mean()
    avg_payoff = payoffs.mean()
    avg_itm_payoff = itm_payoffs.mean() if len(itm_payoffs) > 0 else 0

    print(f"\n📈 {option_type.title()} Option Analysis:")
    print(f"  Strike: ${strike:.0f}")
    print(f"  ITM ratio: {itm_ratio:.1%}")
    print(f"  Average payoff: ${avg_payoff:.2f}")
    print(f"  Average ITM payoff: ${avg_itm_payoff:.2f}")
    print(f"  Max payoff: ${payoffs.max():.2f}")

    return fig


def plot_hedging_performance(
    hedge_pnl: Union[torch.Tensor, np.ndarray],
    underlying_returns: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hedge_name: str = "Hedge Strategy",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Analyze hedging strategy performance.

    Args:
        hedge_pnl: PnL of the hedging strategy
        underlying_returns: Underlying asset returns (optional)
        hedge_name: Name of the strategy for labels
        title: Custom title
        figsize: Figure size
        save_path: Path to save plot

    Returns:
        matplotlib.Figure: The created figure
    """

    # Convert to numpy
    if isinstance(hedge_pnl, torch.Tensor):
        hedge_pnl = hedge_pnl.detach().numpy()

    if underlying_returns is not None and isinstance(underlying_returns, torch.Tensor):
        underlying_returns = underlying_returns.detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # PnL distribution
    ax1 = axes[0]
    ax1.hist(hedge_pnl, bins=25, alpha=0.7, edgecolor="black")
    ax1.axvline(
        hedge_pnl.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: ${hedge_pnl.mean():.0f}",
    )
    ax1.axvline(0, color="black", linestyle="-", alpha=0.5, label="Break-even")
    ax1.set_title(f"{hedge_name} PnL Distribution")
    ax1.set_xlabel("PnL ($)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PnL vs underlying (if provided)
    ax2 = axes[1]
    if underlying_returns is not None:
        scatter_alpha = min(1.0, 100 / len(hedge_pnl))
        ax2.scatter(underlying_returns * 100, hedge_pnl, alpha=scatter_alpha)
        ax2.axhline(0, color="black", linestyle="-", alpha=0.5)
        ax2.axvline(0, color="black", linestyle="-", alpha=0.5)
        ax2.set_title("PnL vs Underlying Return")
        ax2.set_xlabel("Underlying Return (%)")
        ax2.set_ylabel("Hedge PnL ($)")
    else:
        # Show cumulative PnL if no underlying returns
        cumulative_pnl = np.cumsum(hedge_pnl)
        ax2.plot(cumulative_pnl, linewidth=2)
        ax2.axhline(0, color="black", linestyle="-", alpha=0.5)
        ax2.set_title("Cumulative PnL")
        ax2.set_xlabel("Path Index")
        ax2.set_ylabel("Cumulative PnL ($)")
    ax2.grid(True, alpha=0.3)

    # Risk metrics
    ax3 = axes[2]

    # Calculate percentiles for VaR
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pnl_percentiles = np.percentile(hedge_pnl, percentiles)

    ax3.barh(range(len(percentiles)), pnl_percentiles, alpha=0.7)
    ax3.set_yticks(range(len(percentiles)))
    ax3.set_yticklabels([f"{p}%" for p in percentiles])
    ax3.axvline(0, color="black", linestyle="-", alpha=0.5)
    ax3.set_title("PnL Percentiles (VaR)")
    ax3.set_xlabel("PnL ($)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Print performance statistics
    win_rate = (hedge_pnl > 0).mean()
    sharpe = hedge_pnl.mean() / hedge_pnl.std() if hedge_pnl.std() > 0 else 0
    var_95 = np.percentile(hedge_pnl, 5)
    max_loss = hedge_pnl.min()

    print(f"\n📊 {hedge_name} Performance:")
    print(f"  Mean PnL: ${hedge_pnl.mean():.2f}")
    print(f"  PnL Std: ${hedge_pnl.std():.2f}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  VaR (95%): ${var_95:.2f}")
    print(f"  Max Loss: ${max_loss:.2f}")

    return fig


def plot_volatility_analysis(
    instrument,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Analyze volatility patterns for any instrument.

    Args:
        instrument: Any instrument with .volatility attribute
        title: Custom title
        figsize: Figure size
        save_path: Path to save plot

    Returns:
        matplotlib.Figure: The created figure
    """

    if not hasattr(instrument, "volatility"):
        raise ValueError("Instrument must have a 'volatility' attribute")

    vol = instrument.volatility.detach().numpy()
    spot = instrument.spot.detach().numpy()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Volatility over time (first path)
    ax1 = axes[0]
    if hasattr(instrument, "dt"):
        time_axis = np.linspace(
            0, instrument.dt * (vol.shape[1] - 1) * 365, vol.shape[1]
        )
        time_label = "Days"
    else:
        time_axis = np.arange(vol.shape[1])
        time_label = "Time Steps"

    ax1.plot(time_axis, vol[0] * 100, linewidth=2)
    ax1.set_title("Volatility Over Time")
    ax1.set_xlabel(time_label)
    ax1.set_ylabel("Volatility (%)")
    ax1.grid(True, alpha=0.3)

    # Volatility distribution
    ax2 = axes[1]
    vol_flat = vol.flatten()
    ax2.hist(vol_flat * 100, bins=30, alpha=0.7, edgecolor="black")
    ax2.axvline(
        vol_flat.mean() * 100,
        color="red",
        linestyle="--",
        label=f"Mean: {vol_flat.mean():.1%}",
    )
    ax2.set_title("Volatility Distribution")
    ax2.set_xlabel("Volatility (%)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Volatility vs Price Level
    ax3 = axes[2]
    spot_flat = spot.flatten()
    vol_flat = vol.flatten()

    scatter_alpha = min(1.0, 1000 / len(spot_flat))
    ax3.scatter(spot_flat, vol_flat * 100, alpha=scatter_alpha)
    ax3.set_title("Volatility vs Price Level")
    ax3.set_xlabel("Spot Price ($)")
    ax3.set_ylabel("Volatility (%)")
    ax3.grid(True, alpha=0.3)

    # Realized vs Implied (if available)
    ax4 = axes[3]
    if hasattr(instrument, "sigma"):  # For Brownian models
        target_vol = instrument.sigma
        ax4.axhline(
            target_vol * 100,
            color="red",
            linestyle="--",
            label=f"Target: {target_vol:.1%}",
        )
        ax4.plot(time_axis, vol[0] * 100, linewidth=2, label="Realized")
        ax4.set_title("Realized vs Target Volatility")
        ax4.set_xlabel(time_label)
        ax4.set_ylabel("Volatility (%)")
        ax4.legend()
    else:
        # Show volatility term structure
        mean_vol_path = vol.mean(axis=0)
        std_vol_path = vol.std(axis=0)

        ax4.plot(time_axis, mean_vol_path * 100, linewidth=2, label="Mean")
        ax4.fill_between(
            time_axis,
            (mean_vol_path - std_vol_path) * 100,
            (mean_vol_path + std_vol_path) * 100,
            alpha=0.3,
            label="±1 Std",
        )
        ax4.set_title("Volatility Term Structure")
        ax4.set_xlabel(time_label)
        ax4.set_ylabel("Volatility (%)")
        ax4.legend()

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# Convenience function for quick analysis
def quick_instrument_analysis(
    instrument,
    strike: Optional[float] = None,
    option_type: str = "call",
    n_paths_to_show: int = 10,
    save_plots: bool = False,
    output_dir: str = "plots",
) -> None:
    """
    Run complete analysis suite for any instrument.

    Args:
        instrument: Any instrument in our framework
        strike: Strike for option analysis (uses ATM if None)
        option_type: 'call' or 'put'
        n_paths_to_show: Number of paths to show
        save_plots: Whether to save plots
        output_dir: Directory to save plots
    """

    instrument_name = instrument.__class__.__name__

    if save_plots:
        import os

        os.makedirs(output_dir, exist_ok=True)

    # Price path analysis
    print(f"📈 Analyzing {instrument_name}...")
    fig1 = plot_price_paths(
        instrument, n_paths_to_show=n_paths_to_show, title=f"{instrument_name} Analysis"
    )
    if save_plots:
        plt.savefig(f"{output_dir}/{instrument_name}_price_paths.png")
    plt.show()

    # Option analysis
    if strike is None:
        strike = instrument.spot[:, 0].mean().item()

    fig2 = plot_option_analysis(
        instrument,
        strike=strike,
        option_type=option_type,
        title=f"{instrument_name} Option Analysis",
    )
    if save_plots:
        plt.savefig(f"{output_dir}/{instrument_name}_option_analysis.png")
    plt.show()

    # Volatility analysis (if available)
    if hasattr(instrument, "volatility"):
        fig3 = plot_volatility_analysis(
            instrument, title=f"{instrument_name} Volatility Analysis"
        )
        if save_plots:
            plt.savefig(f"{output_dir}/{instrument_name}_volatility_analysis.png")
        plt.show()

    print(f"✅ {instrument_name} analysis complete!")


def plot_hedge_comparison(
    deep_hedge_positions: Union[torch.Tensor, np.ndarray],
    bs_delta: Union[torch.Tensor, np.ndarray],
    deep_hedge_pnl: Union[torch.Tensor, np.ndarray],
    bs_hedge_pnl: Union[torch.Tensor, np.ndarray],
    spots: Union[torch.Tensor, np.ndarray],
    strike: float,
    training_history: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
    performance_results: Optional[dict] = None,
    path_idx: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 14),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create comprehensive hedge comparison visualization.

    Shows training history, hedging positions over time, and final PnL distribution
    comparing deep hedging vs Black-Scholes strategies.

    Args:
        deep_hedge_positions: Deep hedging positions over time, shape (n_paths, n_steps)
        bs_delta: Black-Scholes delta over time, shape (n_paths, n_steps)
        deep_hedge_pnl: Deep hedging cumulative PnL, shape (n_paths, n_steps)
        bs_hedge_pnl: Black-Scholes cumulative PnL, shape (n_paths, n_steps)
        spots: Spot prices over time, shape (n_paths, n_steps)
        strike: Strike price for the option
        training_history: Training loss history (optional)
        performance_results: Performance metrics dict from compare_hedge_performance (optional)
        path_idx: Which path to show for time series plots
        title: Custom title (optional)
        figsize: Figure size
        save_path: Path to save plot (optional)

    Returns:
        matplotlib.Figure: The created figure

    Examples:
        >>> # Basic usage
        >>> fig = plot_hedge_comparison(
        ...     deep_positions, bs_delta, deep_pnl, bs_pnl, spots, strike=50000
        ... )
        >>> plt.show()

        >>> # With training history
        >>> fig = plot_hedge_comparison(
        ...     deep_positions, bs_delta, deep_pnl, bs_pnl, spots,
        ...     strike=50000, training_history=history
        ... )
    """

    # Convert tensors to numpy
    if isinstance(deep_hedge_positions, torch.Tensor):
        deep_hedge_positions = deep_hedge_positions.detach().numpy()
    if isinstance(bs_delta, torch.Tensor):
        bs_delta = bs_delta.detach().numpy()
    if isinstance(deep_hedge_pnl, torch.Tensor):
        deep_hedge_pnl = deep_hedge_pnl.detach().numpy()
    if isinstance(bs_hedge_pnl, torch.Tensor):
        bs_hedge_pnl = bs_hedge_pnl.detach().numpy()
    if isinstance(spots, torch.Tensor):
        spots = spots.detach().numpy()
    if training_history is not None and isinstance(training_history, torch.Tensor):
        training_history = training_history.detach().numpy()

    # Create subplots (3 or 2 depending on whether we have training history)
    n_plots = 3 if training_history is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

    if n_plots == 2:
        ax_hedge, ax_pnl = axes
    else:
        ax_hist, ax_hedge, ax_pnl = axes

    # Plot 1: Training History (if provided)
    if training_history is not None:
        ax_hist.plot(training_history, label="Deep Hedging Training", linewidth=2)
        ax_hist.set_title("Training History", fontsize=14, fontweight="bold")
        ax_hist.set_xlabel("Epoch")
        ax_hist.set_ylabel("Loss")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

    # Plot 2: Hedging Positions Over Time
    ax_spot = ax_hedge.twinx()

    # Show hedging positions for specified path
    time_steps = np.arange(deep_hedge_positions.shape[1])

    # Only plot deep hedge if not NaN
    if not np.isnan(deep_hedge_positions[path_idx]).all():
        ax_hedge.plot(
            time_steps,
            deep_hedge_positions[path_idx],
            "b-",
            linewidth=2,
            label="Deep Hedge",
            alpha=0.8,
        )

    ax_hedge.plot(
        time_steps,
        bs_delta[path_idx],
        "r--",
        linewidth=2,
        label="Black-Scholes",
        alpha=0.8,
    )
    ax_hedge.set_xlabel("Time Step")
    ax_hedge.set_ylabel("Hedge Position", color="black")
    ax_hedge.set_title(
        f"Dynamic Hedging Strategy Over Time (Path {path_idx})",
        fontsize=14,
        fontweight="bold",
    )
    ax_hedge.legend(loc="upper left")
    ax_hedge.grid(True, alpha=0.3)

    # Add spot price on secondary axis
    ax_spot.plot(
        time_steps,
        spots[path_idx],
        "grey",
        linestyle=":",
        linewidth=2,
        alpha=0.6,
        label="Spot Price",
    )
    ax_spot.axhline(strike, color="red", linestyle="--", alpha=0.4, label="Strike")
    ax_spot.set_ylabel("Price ($)", color="grey")
    ax_spot.legend(loc="upper right")

    # Plot 3: Final PnL Distribution
    # Get final PnL
    deep_final_pnl = (
        deep_hedge_pnl[:, -1] if deep_hedge_pnl.ndim == 2 else deep_hedge_pnl
    )
    bs_final_pnl = bs_hedge_pnl[:, -1] if bs_hedge_pnl.ndim == 2 else bs_hedge_pnl

    # Get means from performance_results if provided, otherwise calculate
    if performance_results:
        deep_mean = performance_results.get("Deep Hedge", {}).get(
            "mean", deep_final_pnl.mean()
        )
        bs_mean = performance_results.get("Black-Scholes", {}).get(
            "mean", bs_final_pnl.mean()
        )
    else:
        deep_mean = deep_final_pnl.mean()
        bs_mean = bs_final_pnl.mean()

    # Only plot if data is valid (not NaN)
    if not np.isnan(deep_final_pnl).all():
        ax_pnl.hist(
            deep_final_pnl,
            bins=30,
            alpha=0.5,
            color="blue",
            label="Deep Hedge PnL",
            density=False,
        )
        ax_pnl.axvline(
            deep_mean,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Deep Mean: ${deep_mean:.2f}",
        )

    if not np.isnan(bs_final_pnl).all():
        ax_pnl.hist(
            bs_final_pnl, bins=30, alpha=0.5, color="red", label="BS PnL", density=False
        )
        ax_pnl.axvline(
            bs_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"BS Mean: ${bs_mean:.2f}",
        )

    ax_pnl.set_xlabel("Final PnL ($)")
    ax_pnl.set_ylabel("Frequency")
    pnl_title = title or "Final PnL Distribution"
    if np.isnan(deep_final_pnl).all():
        pnl_title += " (Deep Hedge: NaN - Training Failed)"
    ax_pnl.set_title(pnl_title, fontsize=14, fontweight="bold")
    ax_pnl.legend()
    ax_pnl.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
