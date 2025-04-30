import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import Snowball
from pfhedge.instruments.primary.fixed import FixedStock
from pfhedge.nn.functional import cum_pl
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
    test_n_paths = 1
    mc_n_path = 100000
    loss = ExpectedShortfall(0.5)
    fig_lable = "ESF_4"

    # Setup underlier and derivative
    init_state = 100.0
    dt = 1 / 250
    sigma = 0.21
    mu = 0.0
    cost = 1e-4
    obs_days = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 230, 250]
    notional = 100
    knockout_barrier = 100

    init_spot = torch.tensor(init_state, requires_grad=True)

    # load history quotes
    quotes = pd.read_csv("examples/000905.csv")
    series = torch.tensor(quotes["Close"].values, dtype=torch.float)[:-65]
    fixed_stock = FixedStock(spots=list(series), cost=cost, dt=dt)
    his_vol = fixed_stock.volatility
    print(f"Historic volatility: {his_vol}")

    # Define hedger and features
    features = [
        "log_moneyness",
        "expiry_time",
        "volatility",
        "knocked_in",
        "knockin_barrier_to_strike",
        "prev_hedge",
    ]

    def snowball(ki_barrier, knocked_in=False, underlier=None, offset=0):
        if underlier is None:
            underlier = BrownianStock(sigma=sigma, mu=mu, dt=dt, cost=cost)
        obs = [d - offset for d in obs_days if d >= offset]
        ko_coupons = [0.2 * d / 250 for d in obs_days if d >= offset]
        return Snowball(
            underlier=underlier,
            notional=notional,
            init_spot=init_state,
            strike=100.0,
            maturity=obs[-1] * dt,
            knockin_barrier=ki_barrier,
            observations=[d * dt for d in obs],
            knockout_barriers=[knockout_barrier] * len(obs),
            knockout_coupons=ko_coupons,
            no_touch_coupon=0.05,
            is_knocked_in=knocked_in,
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

    # test hedger
    def test_hedger(hedger: Hedger, ki_barrier):
        with torch.no_grad():
            torch.manual_seed(test_seed)
            derivative_test = snowball(
                ki_barrier, underlier=FixedStock(spots=list(series), cost=cost, dt=dt)
            )
            derivative_test.simulate(n_paths=test_n_paths, init_state=(init_spot,))
            hedging = hedger.compute_hedge(derivative_test).squeeze()
            spots = derivative_test.ul().spot.squeeze()
            pnl = hedger.compute_cum_pl(derivative_test).squeeze()
            return hedging.detach().numpy(), spots.detach().numpy(), pnl.detach().numpy()

    hedgeings, spots, pnls = zip(*[test_hedger(h, 80) for h in hedgers])

    # delta hedge
    def calc_pv(derivative, spot) -> float:
        with torch.no_grad():
            derivative.simulate(n_paths=mc_n_path, init_state=(spot,))
            return derivative.payoff().mean().item()

    def calc_delta(derivative, spot) -> float:
        with torch.no_grad():
            spot_up = spot * 1.01
            spot_down = spot * 0.99
            derivative.simulate(n_paths=mc_n_path, init_state=(spot_up,))
            pv_up = derivative.payoff().mean().item()
            derivative.simulate(n_paths=mc_n_path, init_state=(spot_down,))
            pv_down = derivative.payoff().mean().item()
            return (pv_up - pv_down) / (spot_up - spot_down)

    derivative_test = snowball(80, underlier=FixedStock(spots=list(series), cost=cost, dt=dt))
    derivative_test.simulate(n_paths=test_n_paths, init_state=(init_spot,))
    derivative_test_payoff = derivative_test.payoff()

    delta_hedging = []
    pvs = []
    progress = tqdm(range(obs_days[-1]))
    is_knocked_in = False
    is_knocked_out = False
    for i in progress:
        derivative = snowball(80, offset=i, knocked_in=is_knocked_in)
        s = spots[0][i]
        if s <= 80:
            is_knocked_in = True
        if is_knocked_out or (i in obs_days and s > knockout_barrier):
            is_knocked_out = True
            pvs.append(derivative_test_payoff.item())
            delta_hedging.append(0)
        else:
            pv = calc_pv(derivative, s)
            pvs.append(pv)
            delta = calc_delta(derivative, s)
            delta_hedging.append(delta)
        progress.desc = f"Delta={delta:.4f}"
    pvs = pvs + [0]
    delta_hedging = delta_hedging + [0]
    pvs = np.array(pvs)
    delta_hedging = np.array(delta_hedging)
    # delta hedge pnl
    delta_hedge_pnl = (
        cum_pl(
            derivative_test.ul().spot.unsqueeze(0),
            torch.tensor(delta_hedging).unsqueeze(0).unsqueeze(0),
            cost=[cost],
            payoff=derivative_test_payoff,
        )
        .squeeze()
        .detach()
        .numpy()
    )

    # combine hedging pnl and option pv
    delta_hedge_total_pnl = delta_hedge_pnl - pvs
    total_pnls = [pnl - pvs for pnl in pnls]

    # plot histories and hedgings as subplots
    _, (ax_hist, ax_hedge, ax_pnl) = plt.subplots(3, 1, figsize=(10, 18))
    # training history
    for history, label in zip(histories, labels):
        ax_hist.plot(history, label=label)
    ax_hist.legend()
    ax_hist.set_title("Training History")
    ax_hist.set_xlabel("Epoch")
    ax_hist.set_ylabel("Loss")
    ax_hist.grid(True)
    # hedging comparison
    ax_spot = ax_hedge.twinx()
    ax_hedge.plot(delta_hedging, label="Delta Hedge", color="black")
    for hedge, label in zip(hedgeings, labels):
        ax_hedge.plot(hedge, label=label)
    for spot in spots:
        ax_spot.plot(spot, color="grey", linestyle="--")
    ax_spot.axhline(80, color="red", linestyle="dashed")
    ax_spot.axhline(100, color="red", linestyle="dashed")
    ax_hedge.set_title("Hedging Position")
    ax_hedge.set_xlabel("Day")
    ax_hedge.set_ylabel("Hedging Position")
    ax_spot.set_ylabel("Spot")
    ax_hedge.legend()
    ax_hedge.grid(True)
    # plot hedging pnls
    ax_pnl.plot(delta_hedge_total_pnl, label="Delta Hedge", color="black")
    for pnl, label in zip(total_pnls, labels):
        ax_pnl.plot(pnl, label=label)
    ax_spot_pnl = ax_pnl.twinx()
    for spot in spots:
        ax_spot_pnl.plot(spot, color="grey", linestyle="--")
    ax_spot_pnl.axhline(80, color="red", linestyle="dashed")
    ax_spot_pnl.axhline(100, color="red", linestyle="dashed")
    ax_pnl.set_title("Total PnL")
    ax_pnl.set_xlabel("Day")
    ax_pnl.set_ylabel("Total PnL")
    ax_spot_pnl.set_ylabel("Spot")
    ax_pnl.legend()
    ax_pnl.grid(True)
    plt.tight_layout()
    output_file = f"examples/output/snowball_hedge/result_{fig_lable}_{total_n_epochs}.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved figure to {output_file}")


if __name__ == "__main__":
    main()
