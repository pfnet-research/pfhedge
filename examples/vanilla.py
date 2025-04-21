import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pfhedge.instruments.derivative.european import EuropeanOption
from pfhedge.instruments.primary.brownian import BrownianStock
from pfhedge.nn.modules.hedger import Hedger
from pfhedge.nn.modules.loss import ExpectedShortfall
from pfhedge.nn.modules.mlp import MultiLayerPerceptron


def fit(derivative, loss_fn, n_epochs=200, n_paths=10000, init_state=(1,)):
    # Set default features if not provided
    features = ["log_moneyness", "expiry_time", "volatility", "prev_hedge"]
    # Set default model if not provided
    model = MultiLayerPerceptron(n_layers=4, n_units=[64, 32, 32, 16])
    # Create hedger
    hedger = Hedger(model, features, criterion=loss_fn)
    # Fit hedger
    history = hedger.fit(derivative, n_epochs=n_epochs, n_paths=n_paths, init_state=init_state)
    return hedger, history, init_state


# Example usage of the fit function
if __name__ == "__main__":
    torch.manual_seed(42)

    sigma = 0.2
    mu = 0.0
    cost = 1e-4
    init_spot = 1.0
    n_epochs = 500
    n_paths = 5000

    # Create loss function
    loss = ExpectedShortfall(0.5)

    # Fit hedger
    derivative1 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=1 * init_spot,
        maturity=30 / 250,
    )
    hedger1, history1, init_state1 = fit(
        derivative=derivative1,
        loss_fn=loss,
        n_epochs=n_epochs,
        n_paths=n_paths,
        init_state=(init_spot,),
    )
    derivative2 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=1.03 * init_spot,
        maturity=30 / 250,
    )
    hedger2, history2, init_state2 = fit(
        derivative=derivative2,
        loss_fn=loss,
        n_epochs=n_epochs,
        n_paths=n_paths,
        init_state=(init_spot,),
    )
    derivative3 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=1 * init_spot,
        maturity=15 / 250,
    )
    hedger3, history3, init_state3 = fit(
        derivative=derivative3,
        loss_fn=loss,
        n_epochs=n_epochs,
        n_paths=n_paths,
        init_state=(init_spot,),
    )
    derivative4 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=1.03 * init_spot,
        maturity=15 / 250,
    )
    hedger4, history4, init_state4 = fit(
        derivative=derivative4,
        loss_fn=loss,
        n_epochs=n_epochs,
        n_paths=n_paths,
        init_state=(init_spot,),
    )
    derivative5 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=100 * init_spot,
        maturity=30 / 250,
    )
    hedger5, history5, init_state5 = fit(
        derivative=derivative5,
        loss_fn=loss,
        n_epochs=n_epochs,
        n_paths=n_paths,
        init_state=(init_spot * 100,),
    )
    # plot histories
    plt.plot(history1, label="case 1")
    plt.plot(history2, label="case 2")
    plt.plot(history3, label="case 3")
    plt.plot(history4, label="case 4")
    plt.plot(history5, label="case 5")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("History of Loss")
    plt.savefig("examples/output/vanilla/fithistory.png")
    plt.close()

    # Test performance of hedgers on different derivatives
    hedgers = [hedger1, hedger2, hedger3, hedger4, hedger5]
    derivatives = [derivative1, derivative2, derivative3, derivative4, derivative5]
    init_states = [init_state1, init_state2, init_state3, init_state4, init_state5]
    histories = [history1, history2, history3, history4, history5]
    for d, init_state in zip(derivatives, init_states):
        d.simulate(n_paths=n_paths, init_state=init_state)
    # Compute loss
    losses = [
        [
            h.compute_loss(d, n_paths=n_paths, init_state=init_state, enable_grad=False).item()
            for h in hedgers
        ]
        for d, init_state in zip(derivatives, init_states)
    ]
    df = pd.DataFrame(
        losses,
        columns=["hedger1", "hedger2", "hedger3", "hedger4", "hedger5"],
        index=["derivative1", "derivative2", "derivative3", "derivative4", "derivative5"],
    )
    print(df)
    df.to_csv("examples/output/vanilla/loss.csv")
    # Compute PnL
    pnls = [[h.compute_pl(d) for h in hedgers] for d in derivatives]
    # Plot PnL distributions
    n_rows = len(pnls)
    n_cols = len(pnls[0]) if n_rows > 0 else 0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            pnl_np = pnls[i][j].detach().cpu().numpy()
            axes[i][j].hist(pnl_np, bins=50, alpha=0.7, density=True)
            mean = np.mean(pnl_np)
            std = np.std(pnl_np)
            axes[i][j].text(
                0.05,
                0.95,
                f"Mean: {mean:.6f}\nStd: {std:.6f}\nLoss: {histories[i][-1]:.6f}",
                transform=axes[i][j].transAxes,
                verticalalignment="top",
            )
            if i == n_rows - 1:
                axes[i][j].set_xlabel(f"hedger {j + 1}")
            if j == 0:
                axes[i][j].set_ylabel(f"derivative {i + 1}")
    # Add title
    plt.suptitle("PnL Distributions")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig("examples/output/vanilla/pnl.png")
    plt.close()
