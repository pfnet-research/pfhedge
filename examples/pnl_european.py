from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt  # noqa: F401 - Used in commented plotting sections
import numpy as np
import pandas as pd
import torch

from pfhedge.instruments import (
    BrownianStock,
    EuropeanOption,
    FixedStock,
)
from pfhedge.nn import (
    BlackScholes,
    BSEuropeanOption,
    EntropicRiskMeasure,
    ExpectedShortfall,
    Hedger,
    MultiLayerPerceptron,
)
from torch.nn import ReLU


def load_quotes(file_path: str) -> torch.Tensor:
    quotes = pd.read_csv(file_path)
    close = torch.tensor(quotes["Close"].values, dtype=torch.float)
    return close


def generate_log_normal_brownian(
    sigma: float, mu: float, dt: float, n_steps: int, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a log-normal Brownian motion time series.

    Args:
        sigma (float): Volatility parameter.
        dt (float): Time step size.
        n_steps (int): Number of time steps.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the log series and the
            exponential series.
    """
    if seed is not None:
        torch.manual_seed(seed)
    # Generate Brownian increments
    increments = torch.randn(n_steps - 1) * np.sqrt(sigma**2 * dt) + mu * dt
    # Convert to log-normal Brownian motion
    log_series = torch.cat([torch.zeros(1), torch.cumsum(increments, dim=0)])
    series = torch.exp(log_series)
    return series


def get_derivative(
    sigma: float, mu: float, dt: float, strike: float, maturity: float, cost: float = 0.0
) -> EuropeanOption:
    stock = BrownianStock(cost=cost, sigma=sigma, mu=mu, dt=dt)
    derivative = EuropeanOption(stock, strike=strike, maturity=maturity)
    return derivative


def loss_hedger(loss, derivative, features, n_epochs, n_paths, init_spot) -> Hedger:
    # Create MLP hedger
    model = MultiLayerPerceptron(n_layers=4, n_units=[64, 32, 32, 16], activation=ReLU())
    hedger = Hedger(model, features, criterion=loss)
    # Fit the hedger
    history = hedger.fit(derivative, n_epochs=n_epochs, n_paths=n_paths, init_state=(init_spot,))
    return hedger, history


def blackscholes_hedger(derivative) -> Hedger:
    # Create instruments
    model = BlackScholes(derivative)
    hedger_bs = Hedger(model, model.inputs())
    return hedger_bs


torch.manual_seed(1234)

# Set up the instruments
dt = 1 / 250
vol = 0.4
mu = 0.0
init_spot = 100
cost = 0.001

# Set up the derivative
strike = 100
term = 30
maturity = term * dt
features = ["log_moneyness", "expiry_time", "volatility", "prev_hedge"]

# series = generate_log_normal_brownian(sigma=vol, mu=mu, dt=dt, n_steps=term * 3 + 10)
series = load_quotes("examples/000905.csv")
fixed_stock = FixedStock(spots=list(series), cost=cost, dt=dt)
sigma = fixed_stock.volatility  # use realized volatility in simulation
print(f"Realized Volatility: {sigma:.6f}")

# Set trainging parameters
n_epochs = 10
n_paths = 1000

# Set up the hedgers
loss_functions = {
    "Entropic Risk Measure (a=1)": EntropicRiskMeasure(a=1),
    "Entropic Risk Measure (a=0.1)": EntropicRiskMeasure(a=0.1),
    "Expected Shortfall (95%)": ExpectedShortfall(0.05),
    "Expected Shortfall (50%)": ExpectedShortfall(0.5),
}
hedgers = {}
derivative = get_derivative(sigma, mu=mu, dt=dt, cost=cost, strike=strike, maturity=maturity)
hedger = blackscholes_hedger(derivative)
hedgers["Black-Scholes"] = (hedger, derivative, [])
for name, loss in loss_functions.items():
    derivative = get_derivative(sigma, mu=mu, dt=dt, cost=cost, strike=strike, maturity=maturity)
    hedger, history = loss_hedger(loss, derivative, features, n_epochs, n_paths, init_spot)
    hedgers[name] = (hedger, derivative, history)

# PnL statistics
pnls = {}
histories = {}
for name, (hedger, derivative, history) in hedgers.items():
    derivative.simulate(n_paths=1000, init_state=(init_spot,))
    pnls[name] = hedger.compute_pl(derivative)
    histories[name] = history
print("PnL Statistics:")
for name, pnl in pnls.items():
    print(f"{name}: {pnl.mean():.6f} ({pnl.std():.6f})")

# Calculate global min and max for uniform binning
all_pnls = torch.cat([pnl for pnl in pnls.values()]).detach().numpy()
bin_edges = np.linspace(all_pnls.min(), all_pnls.max(), 101)

# Plot PnL distributions and fitting history in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# Plot PnL distributions
for name, pnl in pnls.items():
    ax1.hist(pnl.detach().numpy(), bins=bin_edges, label=name, histtype="step", alpha=0.5)
ax1.set_title("PnL Distribution")
ax1.set_xlabel("PnL")
ax1.set_ylabel("Frequency")
ax1.legend()
ax1.grid(True)
# Plot fitting history
for name, history in histories.items():
    ax2.plot(history, label=name)
ax2.set_title("Training History")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_yscale('log')  # Add log scale
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()

# Backtest on the fixed stock
fixed_derivative = EuropeanOption(fixed_stock, strike=strike, maturity=maturity)
fixed_derivative.simulate(n_paths=fixed_stock.max_n_paths(maturity), init_state=(init_spot,))
backtest_pnls = {}
for name, (hedger, _) in hedgers.items():
    pnl = hedger.compute_pl(fixed_derivative)
    backtest_pnls[name] = pnl
    print(f"{name}: {pnl.mean():.6f} ({pnl.std():.6f})")

# calculate min and max for uniform binning
all_pnls = torch.cat([pnl for pnl in backtest_pnls.values()]).detach().numpy()
bin_edges = np.linspace(all_pnls.min(), all_pnls.max(), 101)
# plot backtest pnl
plt.figure(figsize=(12, 8))
for name, pnl in backtest_pnls.items():
    plt.hist(pnl.detach().numpy(), bins=bin_edges, label=name, histtype="step", alpha=0.5)
plt.title("PnL Distribution (Backtest)")
plt.xlabel("PnL")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# daily PnL
bs_priceable = BSEuropeanOption.from_derivative(fixed_derivative)
daily_pnls = {}
# select paths
daily_pnl_num = 10
pnl_days = torch.linspace(0, fixed_stock.max_n_paths(maturity) - 1, daily_pnl_num).long()
for name, (hedger, _) in hedgers.items():
    daily_pnls[name] = hedger.compute_cum_pl(fixed_derivative, priceable=bs_priceable)[pnl_days]

# Get corresponding stock paths
stock_paths = fixed_stock.spot[pnl_days]

# Create subplot grid based on path count
n_cols = 2
n_rows = (daily_pnl_num + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
colors = plt.cm.tab10.colors
# Plot each path across all strategies
for path_idx, ax in enumerate(axes.flat):
    if path_idx >= daily_pnl_num:
        ax.set_visible(False)
        continue

    # Plot strategies' PnL
    for i, (name, pnl) in enumerate(daily_pnls.items()):
        pnl_path = pnl[path_idx].detach().numpy()
        color = colors[i % len(colors)]
        ax.plot(pnl_path, color=color, label=name, alpha=0.7)

    # Create twin axis for stock price
    ax2 = ax.twinx()
    stock_price = stock_paths[path_idx].squeeze().detach().numpy()
    ax2.plot(stock_price, color='#444444', linestyle='--', alpha=0.4, label='Stock Price')
    ax2.set_ylabel('Stock Price', color='#444444')
    ax2.tick_params(axis='y', labelcolor='#444444')

    ax.set_title(f"Path {pnl_days[path_idx].item()} (K={fixed_derivative.strike:.1f})")
    ax.set_xlabel("Days")
    ax.set_ylabel("PnL")
    ax.grid(True)
    # Add legend to first subplot
    if path_idx == 0:
        ax.legend(framealpha=0.9)
plt.suptitle("Strategy PnL Comparison Across Paths")
plt.tight_layout()
plt.show()
