import matplotlib.pyplot as plt
import numpy as np
import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import Snowball
from pfhedge.nn.modules.hedger import Hedger
from pfhedge.nn.modules.loss import ExpectedShortfall
from pfhedge.nn.modules.mlp import MultiLayerPerceptron


def main():
    torch.manual_seed(42)
    n_paths = 1000
    n_epochs = 100
    loss = ExpectedShortfall(0.5)

    init_spot = 100.0
    dt = 1 / 250
    stock = BrownianStock(sigma=0.2, mu=0.0, dt=dt, cost=0.0)
    obs_days = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 230, 250]

    derivative_lin = Snowball(
        underlier=stock,
        notional=1.0,
        init_spot=init_spot,
        strike=100.0,
        maturity=obs_days[-1] * dt,
        knockin_barrier=lambda x: torch.linspace(70, 90, x),
        observations=[d * dt for d in obs_days],
        knockout_barriers=[100.0] * len(obs_days),
        knockout_coupons=[0.2 * d / 250 for d in obs_days],  # Higher coupons as compensation
        no_touch_coupon=0.05,
        is_knocked_in=False,
    )

    features = [
        "log_moneyness",
        "expiry_time",
        "volatility",
        "knocked_in",
        "knockin_barrier_to_strike",
        "prev_hedge",
    ]

    hedger_lin = Hedger(
        MultiLayerPerceptron(n_layers=4, n_units=[64, 32, 32, 16]),  # Larger network
        features,
        criterion=loss,
    )

    # Fit hedger
    history_lin = hedger_lin.fit(
        derivative_lin, n_epochs=n_epochs, n_paths=n_paths, init_state=(init_spot,)
    )

    torch.manual_seed(42)
    derivative_rand = Snowball(
        underlier=stock,
        notional=1.0,
        init_spot=init_spot,
        strike=100.0,
        maturity=obs_days[-1] * dt,
        knockin_barrier=lambda x: torch.rand(x) * 20 + 70,
        observations=[d * dt for d in obs_days],
        knockout_barriers=[100.0] * len(obs_days),
        knockout_coupons=[0.2 * d / 250 for d in obs_days],  # Higher coupons as compensation
        no_touch_coupon=0.05,
        is_knocked_in=False,
    )

    features = [
        "log_moneyness",
        "expiry_time",
        "volatility",
        "knocked_in",
        "knockin_barrier_to_strike",
        "prev_hedge",
    ]

    hedger_rand = Hedger(
        MultiLayerPerceptron(n_layers=4, n_units=[64, 32, 32, 16]),  # Larger network
        features,
        criterion=loss,
    )

    # Fit hedger
    history_rand = hedger_rand.fit(
        derivative_rand, n_epochs=n_epochs, n_paths=n_paths, init_state=(init_spot,)
    )

    torch.manual_seed(42)
    derivative_fixed = Snowball(
        underlier=stock,
        notional=1.0,
        init_spot=init_spot,
        strike=100.0,
        maturity=obs_days[-1] * dt,
        knockin_barrier=80,
        observations=[d * dt for d in obs_days],
        knockout_barriers=[100.0] * len(obs_days),
        knockout_coupons=[0.2 * d / 250 for d in obs_days],  # Higher coupons as compensation
        no_touch_coupon=0.05,
        is_knocked_in=False,
    )

    features = [
        "log_moneyness",
        "expiry_time",
        "volatility",
        "knocked_in",
        "prev_hedge",
    ]

    hedger_fixed = Hedger(
        MultiLayerPerceptron(n_layers=4, n_units=[64, 32, 32, 16]),  # Larger network
        features,
        criterion=loss,
    )

    # Fit hedger
    history_fixed = hedger_fixed.fit(
        derivative_fixed, n_epochs=n_epochs, n_paths=n_paths, init_state=(init_spot,)
    )

    # Plot history
    plt.figure(figsize=(10, 6))
    plt.plot(history_lin, label="Linear KI barrier")
    plt.plot(history_rand, label="Random KI barrier")
    plt.plot(history_fixed, label="Fixed KI barrier")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("History of Loss - Barrier Snowball")
    plt.grid(True)
    plt.savefig("examples/output/snowball_barrier/fithistory.png")
    plt.close()

    # PnL distribution
    torch.manual_seed(888)
    hedgers = [hedger_lin, hedger_rand, hedger_fixed]
    losses = [history_lin[-1], history_rand[-1], history_fixed[-1]]
    derivative_fixed.simulate(n_paths=n_paths, init_state=(init_spot,))
    pnls = [h.compute_pl(derivative_fixed) for h in hedgers]
    labels = ["Linear KI barrier", "Random KI barrier", "Fixed KI barrier"]

    # Compute statistics for each PnL distribution
    pnl_stats = []
    for i, pnl in enumerate(pnls):
        pnl_np = pnl.detach().numpy()
        mean = np.mean(pnl_np)
        std = np.std(pnl_np)
        var_95 = np.percentile(pnl_np, 5)
        pnl_stats.append({"Mean": mean, "Std": std, "VaR(95%)": var_95, "Loss": losses[i]})

    # Print PnL statistics
    print("\nPnL Statistics:")
    for stat in pnl_stats:
        print(
            f"Mean: {stat['Mean']:.6f}, Std: {stat['Std']:.6f},"
            f" VaR(95%): {stat['VaR(95%)']:.6f}, Loss: {stat['Loss']}"
        )

    # Plot PnL distributions
    n_row = int(np.ceil(len(pnls) / 2))
    _, axes = plt.subplots(n_row, 2, figsize=(14, 6 * n_row))
    if n_row == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, pnl in enumerate(pnls):
        pnl_np = pnl.detach().numpy()
        axes[i // 2, i % 2].hist(
            pnl_np, bins=100, histtype="step", density=True, linewidth=2, label=labels[i]
        )
        axes[i // 2, i % 2].axvline(x=0, color="r", linestyle="--", alpha=0.7)
        axes[i // 2, i % 2].set_xlabel("PnL")
        axes[i // 2, i % 2].set_ylabel("Density")
        axes[i // 2, i % 2].set_title("PnL Distribution")
        axes[i // 2, i % 2].legend()
        # Add statistics annotation
        stat_text = (
            f"Mean: {pnl_stats[i]['Mean']:.6f}\nStd: {pnl_stats[i]['Std']:.6f}\n"
            f"VaR(95%): {pnl_stats[i]['VaR(95%)']:.6f}\nLoss: {pnl_stats[i]['Loss']}"
        )
        axes[i // 2, i % 2].annotate(
            stat_text,
            xy=(0.6, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            va="top",
        )

    plt.tight_layout()
    plt.savefig("examples/output/snowball_barrier/pnl.png")
    plt.close()


if __name__ == "__main__":
    main()
