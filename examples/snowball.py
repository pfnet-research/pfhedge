import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import Snowball
from pfhedge.nn.modules.hedger import Hedger
from pfhedge.nn.modules.loss import EntropicRiskMeasure
from pfhedge.nn.modules.mlp import MultiLayerPerceptron


def main():
    torch.manual_seed(42)
    pv_n_paths = 1000000
    fit_n_paths = 10000
    fit_n_epochs = [2, 5, 10, 20, 50, 100]
    loss = EntropicRiskMeasure(a=1)
    prefix = "ERM1"

    init_state = 100.0
    dt = 1 / 250
    stock = BrownianStock(sigma=0.2, mu=0.0, dt=dt, cost=1e-4)
    obs_days = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 230, 250]
    derivative = Snowball(
        underlier=stock,
        notional=1.0,
        init_spot=init_state,
        strike=100.0,
        maturity=obs_days[-1] * dt,
        knockin_barrier=80.0,
        observations=[d * dt for d in obs_days],
        knockout_barriers=[100.0] * len(obs_days),
        knockout_coupons=[0.5 * d / 250 for d in obs_days],
        no_touch_coupon=0.05,
        is_knocked_in=False,
    )

    init_spot = torch.tensor(init_state, requires_grad=True)
    start_time = time.time()
    derivative.simulate(n_paths=pv_n_paths, init_state=(init_spot,))
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time:.6f}")
    start_time = time.time()
    payoff = derivative.payoff()
    end_time = time.time()
    print(f"Payoff time: {end_time - start_time:.6f}")
    start_time = time.time()
    pv = payoff.mean()
    stddev = payoff.std()
    end_time = time.time()
    print(f"Average time: {end_time - start_time:.6f}")
    print(f"PV: {pv.item()} ({stddev.item() / sqrt(pv_n_paths)})")

    # # Compute the derivative of mean payoff with respect to initial spot
    # start_time = time.time()
    # pv.backward()
    # delta = init_spot.grad.item()
    # end_time = time.time()
    # print(f"Autograd time: {end_time - start_time:.6f}")
    # print(f"Delta (autograd): {delta}")

    # # Compute delta using finite difference
    # start_time = time.time()
    # init_spot_up = torch.tensor(100.1)
    # derivative.simulate(n_paths=n_paths, init_state=(init_spot_up,))
    # payoff_up = derivative.payoff()
    # pv_up = payoff_up.mean()
    # stddev_up = payoff_up.std().item()
    # init_spot_down = torch.tensor(99.9)
    # derivative.simulate(n_paths=n_paths, init_state=(init_spot_down,))
    # payoff_down = derivative.payoff()
    # pv_down = payoff_down.mean()
    # stddev_down = payoff_down.std().item()
    # delta_fd = (pv_up - pv_down) / 0.2
    # end_time = time.time()
    # print(f"Finite difference time: {end_time - start_time:.6f}")
    # print(f"PV_up: {pv_up.item()} ({stddev_up / sqrt(n_paths)})")
    # print(f"PV_down: {pv_down.item()} ({stddev_down / sqrt(n_paths)})")
    # print(f"Delta (fd): {delta_fd.item()}")

    # Create hedger
    features = ["log_moneyness", "expiry_time", "volatility", "knocked_in", "prev_hedge"]
    hedger = Hedger(
        MultiLayerPerceptron(n_layers=4, n_units=[32, 32, 32, 32]),
        features,
        criterion=loss,
    )
    # Fit hedger
    history, hedgers = hedger.fit(
        derivative,
        n_epochs=fit_n_epochs[-1],
        n_paths=fit_n_paths,
        init_state=(init_spot,),
        snapshots=fit_n_epochs[:-1],
    )
    # plot history
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("History of Loss")
    plt.savefig(f"examples/output/snowball/{prefix}_fithistory.png")
    plt.close()

    # Price from hedging
    price = hedger.price(derivative, n_paths=pv_n_paths, init_state=(init_spot,))
    print(f"Price: {price.item()}")

    # PnL distribution
    derivative.simulate(n_paths=fit_n_paths, init_state=(init_spot,))
    pnls = [h.compute_pl(derivative) for h in hedgers]

    n_row = int(np.ceil(len(pnls) / 2))
    _, axes = plt.subplots(n_row, 2, figsize=(12, 5 * n_row))
    # # Define bin edges based on both distributions
    # min_val = torch.cat([pnl0, pnl1]).min().detach().numpy()
    # max_val = torch.cat([pnl0, pnl1]).max().detach().numpy()
    # bin_edges = np.linspace(min_val, max_val, 101)
    # Plot hedger0 PnL in first subplot
    for i, pnl in enumerate(pnls):
        axes[i // 2, i % 2].hist(pnl.detach().numpy(), bins=100, histtype="step")
        axes[i // 2, i % 2].set_xlabel("PnL")
        axes[i // 2, i % 2].set_ylabel("Frequency")
        axes[i // 2, i % 2].set_title(f"PnL Distribution - {fit_n_epochs[i]} epochs")
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"examples/output/snowball/{prefix}_pnl.png")
    plt.close()


if __name__ == "__main__":
    main()
