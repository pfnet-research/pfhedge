import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import Snowball
from pfhedge.nn.modules.hedger import Hedger
from pfhedge.nn.modules.loss import EntropicRiskMeasure
from pfhedge.nn.modules.loss import ExpectedShortfall
from pfhedge.nn.modules.mlp import MultiLayerPerceptron


def main():
    # Re-train hedger for Snowball
    train_seed = 888
    test_seed = 42
    torch.manual_seed(0)
    n_paths = 1000
    total_n_epochs = 200
    test_n_paths = 10000
    loss = ExpectedShortfall(0.5)
    loss_lable = "ESF0"

    # Setup underlier and derivative
    init_state = 100.0
    dt = 1 / 250
    sigma = 0.2
    mu = 0.0
    cost = 0
    obs_days = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 230, 250]
    notional = 100

    init_spot = torch.tensor(init_state, requires_grad=True)

    # Define hedger and features
    features = [
        "log_moneyness",
        "expiry_time",
        "volatility",
        "knocked_in",
        "knockin_barrier_to_strike",
        "prev_hedge",
    ]

    def snowball(ki_barrier):
        return Snowball(
            underlier=BrownianStock(sigma=sigma, mu=mu, dt=dt, cost=cost),
            notional=notional,
            init_spot=init_state,
            strike=100.0,
            maturity=obs_days[-1] * dt,
            knockin_barrier=ki_barrier,
            observations=[d * dt for d in obs_days],
            knockout_barriers=[100.0] * len(obs_days),
            knockout_coupons=[0.2 * d / 250 for d in obs_days],
            no_touch_coupon=0.05,
            is_knocked_in=False,
        )

    def hedger(features_):
        return Hedger(
            MultiLayerPerceptron(n_layers=4, n_units=[32, 32, 32, 32]),
            features_,
            criterion=loss,
        )

    # Fit hedger with re-train
    def fit_retrain(n_epochs, n_retrain):
        torch.manual_seed(train_seed)
        history = []
        hedger_retrain = hedger(features)
        ki_barriers = torch.rand(n_retrain) * 20 + 70
        progress = tqdm(range(n_retrain))
        for i in progress:
            derivative = snowball(ki_barriers[i].item())
            h = hedger_retrain.fit(
                derivative,
                n_epochs=n_epochs,
                n_paths=n_paths,
                init_state=(init_spot,),
                verbose=False,
            )

            progress.desc = f"Loss={h[-1]:.4f}"
            history += h
        return hedger_retrain, history

    hedger_retrain_1, history_retrain_1 = fit_retrain(1, total_n_epochs)
    hedger_retrain_10, history_retrain_10 = fit_retrain(10, total_n_epochs // 10)

    # Fit hedger with random barrier
    torch.manual_seed(train_seed)
    derivative_rand = snowball(lambda x: torch.rand(x) * 20 + 70)
    hedger_rand = hedger(features)
    history_rand = hedger_rand.fit(
        derivative_rand, n_epochs=total_n_epochs, n_paths=n_paths, init_state=(init_spot,)
    )

    # Fit hedger with fixed barrier
    torch.manual_seed(train_seed)
    derivative_fixed = snowball(80.0)
    features_fixed = [
        "log_moneyness",
        "expiry_time",
        "volatility",
        "knocked_in",
        "prev_hedge",
    ]
    hedger_fixed = hedger(features_fixed)
    history_fixed = hedger_fixed.fit(
        derivative_fixed, n_epochs=total_n_epochs, n_paths=n_paths, init_state=(init_spot,)
    )

    hedgers = [hedger_retrain_1, hedger_retrain_10, hedger_rand, hedger_fixed]
    histories = [history_retrain_1, history_retrain_10, history_rand, history_fixed]
    labels = ["Re-train (1 epoch)", "Re-train (10 epochs)", "Random Barrier", "Fixed Barrier (80)"]

    # test hedger for different barriers
    def test_hedger(hedger, ki_barrier):
        with torch.no_grad():
            torch.manual_seed(test_seed)
            derivative_test = snowball(ki_barrier)
            derivative_test.simulate(n_paths=test_n_paths, init_state=(init_spot,))
            option_pnl = -derivative_test.payoff()
            hedge_pnl = hedger.compute_portfolio(derivative_test)
            pnl = hedger.compute_pl(derivative_test)
            # diff_max = (pnl - (hedge_pnl + option_pnl)).abs().max()
            # if diff_max > 1e-9:
            #     raise ValueError(f"diff_max: {diff_max}")
            return (
                pnl.detach().numpy(),
                loss(pnl).detach().item(),
                option_pnl.detach().numpy(),
                hedge_pnl.detach().numpy(),
            )

    pnls = {}
    losses = {}
    option_pnls = {}
    hedge_pnls = {}
    pnl_stats = {}
    test_ki_barriers = [75, 80, 85]
    for b in test_ki_barriers:
        res = [test_hedger(h, b) for h in hedgers]
        pnls[b] = [r[0] for r in res]
        losses[b] = [r[1] for r in res]
        option_pnls[b] = [r[2] for r in res]
        hedge_pnls[b] = [r[3] for r in res]

    # Compute statistics for each PnL distribution
    for b in test_ki_barriers:
        pnl_stats[b] = [
            {
                "Mean": np.mean(pnl),
                "Std": np.std(pnl),
                "Loss": l,
            }
            for pnl, l in zip(pnls[b], losses[b])
        ]

    # Plot histories and PnL distributions: 1 history + 3 PnL subplots
    _, axs = plt.subplots(4, 2, figsize=(20, 40))
    axs = axs.flatten()
    # Subplot 1: fitting histories
    for h, l in zip(histories, labels):
        axs[0].plot(h, label=l)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("History of Loss - Snowball")
    axs[0].grid(True)

    # PnL subplots for each barrier
    for idx, b in enumerate(pnls.keys()):
        ax = axs[idx + 1]
        # plot option payoff PnL
        ax.hist(option_pnls[b][0], bins=100, alpha=0.5, label="Option PnL")
        # plot each hedger PnL
        for pnl, lbl in zip(pnls[b], labels):
            ax.hist(
                pnl,
                bins=100,
                histtype="step",
                label=lbl,
            )
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"PnL Distribution - KI Barrier {b}")
        ax.grid(True)
        stats_text = "\n".join(
            [
                f"{labels[i]}: mean={stat['Mean']:.3f}, std={stat['Std']:.3f}, loss={stat['Loss']:.3f}"
                for i, stat in enumerate(pnl_stats[b])
            ]
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # scatter of hedging pnl to option pnl
    for idx, l in enumerate(labels):
        ax = axs[idx + 4]
        ax.scatter(option_pnls[80][idx], hedge_pnls[80][idx])
        ax.set_xlabel("Option PnL")
        ax.set_ylabel("Hedging PnL")
        ax.set_title(f"{l} - KI Barrier 80")
        ax.grid(True)
        # plot reference line x + y = 0 ⇒ y = -x
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        pts = np.array([max(x_min, -y_max), min(x_max, -y_min)])
        ax.plot(pts, -pts, "r--", linewidth=1)

    plt.tight_layout()
    output_file = f"examples/output/snowball_re_train/results_{loss_lable}_{notional}_{train_seed}_{total_n_epochs}.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
