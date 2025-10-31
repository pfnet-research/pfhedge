import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from pfhedge.instruments.derivative.european import EuropeanOption
from pfhedge.instruments.primary.brownian import BrownianStock
from pfhedge.nn.modules.hedger import Hedger
from pfhedge.nn.modules.loss import ExpectedShortfall
from pfhedge.nn.modules.mlp import MultiLayerPerceptron


def fit(derivative, loss_fn, n_epochs=200, n_paths=10000, init_state=(1,), model_path=None):
    # Set default features if not provided
    features = ["log_moneyness", "expiry_time", "volatility", "prev_hedge"]
    # Set default model if not provided
    model = MultiLayerPerceptron(n_layers=4, n_units=[64, 32, 32, 16])
    # Create hedger
    hedger = Hedger(model, features, criterion=loss_fn)

    # Check if model exists and load it
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        hedger.load_state_dict(torch.load(model_path))
        # Return empty history since we didn't train
        history = []
    else:
        print(f"Training new model (no existing model at {model_path})")
        # Fit hedger
        history = hedger.fit(derivative, n_epochs=n_epochs, n_paths=n_paths, init_state=init_state)
        # Save model if path provided
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(hedger.state_dict(), model_path)
            print(f"Saved model to {model_path}")

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
        model_path="/tmp/vanilla_model.pth",
    )
    derivative2 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=1.03 * init_spot,
        maturity=30 / 250,
    )
    # hedger2, history2, init_state2 = fit(
    #     derivative=derivative2,
    #     loss_fn=loss,
    #     n_epochs=n_epochs,
    #     n_paths=n_paths,
    #     init_state=(init_spot,),
    # )
    derivative3 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=1 * init_spot,
        maturity=15 / 250,
    )
    # hedger3, history3, init_state3 = fit(
    #     derivative=derivative3,
    #     loss_fn=loss,
    #     n_epochs=n_epochs,
    #     n_paths=n_paths,
    #     init_state=(init_spot,),
    # )
    derivative4 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=1.03 * init_spot,
        maturity=15 / 250,
    )
    # hedger4, history4, init_state4 = fit(
    #     derivative=derivative4,
    #     loss_fn=loss,
    #     n_epochs=n_epochs,
    #     n_paths=n_paths,
    #     init_state=(init_spot,),
    # )
    derivative5 = EuropeanOption(
        BrownianStock(sigma=sigma, mu=mu, cost=cost),
        call=True,
        strike=100 * init_spot,
        maturity=30 / 250,
    )
    # hedger5, history5, init_state5 = fit(
    #     derivative=derivative5,
    #     loss_fn=loss,
    #     n_epochs=n_epochs,
    #     n_paths=n_paths,
    #     init_state=(init_spot * 100,),
    # )
    # plot histories (only if we actually trained)
    if history1:
        plt.plot(history1, label="case 1")
        # plt.plot(history2, label="case 2")
        # plt.plot(history3, label="case 3")
        # plt.plot(history4, label="case 4")
        # plt.plot(history5, label="case 5")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("History of Loss")
        plt.savefig("/tmp/fithistory.png")
        print("Saved training history to /tmp/fithistory.png")
    else:
        print("Skipped training history plot (loaded existing model)")
    plt.close()

    # Test performance of hedgers on different derivatives
    hedgers = [hedger1]#, hedger2, hedger3, hedger4, hedger5]
    derivatives = [derivative1]# derivative2, derivative3, derivative4, derivative5]
    init_states = [init_state1]#, init_state2, init_state3, init_state4, init_state5]
    histories = [history1]#, history2, history3, history4, history5]
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
        columns=["hedger1"],# "hedger2", "hedger3", "hedger4", "hedger5"],
        index=["derivative1"],# "derivative2", "derivative3", "derivative4", "derivative5"],
    )
    print(df)
    df.to_csv("/tmp/loss.csv")

    # Compute hedging positions
    print("\nComputing hedging positions...")
    hedge_positions = [[h.compute_hedge(d) for h in hedgers] for d in derivatives]

    # Determine grid size
    n_rows = len(hedge_positions)
    n_cols = len(hedge_positions[0]) if n_rows > 0 else 0

    # Save hedge positions to CSV (first path only for readability)
    for i, d in enumerate(derivatives):
        for j, h in enumerate(hedgers):
            positions = hedge_positions[i][j][0].detach().cpu().numpy()  # First path
            # Ensure 1D array
            if positions.ndim > 1:
                positions = positions.flatten()
            pd.DataFrame({
                'time_step': range(len(positions)),
                'hedge_position': positions
            }).to_csv(f"/tmp/hedge_positions_d{i+1}_h{j+1}.csv", index=False)
    print(f"Saved hedge positions to /tmp/hedge_positions_*.csv")

    # Plot hedge positions over time
    print("Plotting hedge positions...")
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i in range(n_rows):
        for j in range(n_cols):
            positions = hedge_positions[i][j].detach().cpu().numpy()  # Shape: (n_paths, n_steps)

            # Ensure 2D: (n_paths, n_steps)
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
            elif positions.ndim > 2:
                positions = positions.reshape(positions.shape[0], -1)

            time_steps = np.arange(positions.shape[1])

            # Plot mean and std bands - ensure 1D
            mean_pos = positions.mean(axis=0).flatten()
            std_pos = positions.std(axis=0).flatten()

            axes2[i][j].plot(time_steps, mean_pos, label='Mean hedge', linewidth=2, color='blue')
            axes2[i][j].fill_between(
                time_steps,
                mean_pos - std_pos,
                mean_pos + std_pos,
                alpha=0.3,
                color='blue',
                label='±1 std'
            )

            # Plot a few sample paths
            for path_idx in range(min(5, positions.shape[0])):
                axes2[i][j].plot(time_steps, positions[path_idx], alpha=0.2, color='gray', linewidth=0.5)

            axes2[i][j].set_xlabel('Time Step')
            axes2[i][j].set_ylabel('Hedge Position (units)')
            axes2[i][j].set_title(f'Derivative {i+1}, Hedger {j+1}')
            axes2[i][j].legend()
            axes2[i][j].grid(True, alpha=0.3)

            # Add statistics text
            final_mean = mean_pos[-1]
            final_std = std_pos[-1]
            axes2[i][j].text(
                0.95, 0.95,
                f'Final: {final_mean:.3f}±{final_std:.3f}\nStd over time: {std_pos.mean():.4f}',
                transform=axes2[i][j].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

    plt.suptitle('Hedging Positions Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/tmp/hedge_positions.png", dpi=150)
    print("Saved hedge position plot to /tmp/hedge_positions.png")
    plt.close()

    # Compute PnL
    pnls = [[h.compute_pl(d) for h in hedgers] for d in derivatives]
    # Plot PnL distributions
    n_rows = len(pnls)
    n_cols = len(pnls[0]) if n_rows > 0 else 0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for i in range(n_rows):
        for j in range(n_cols):
            pnl_np = pnls[i][j].detach().cpu().numpy()
            axes[i][j].hist(pnl_np, bins=50, alpha=0.7, density=True)
            mean = np.mean(pnl_np)
            std = np.std(pnl_np)
            loss_text = f"{histories[i][-1]:.6f}" if histories[i] else "N/A (loaded model)"
            axes[i][j].text(
                0.05,
                0.95,
                f"Mean: {mean:.6f}\nStd: {std:.6f}\nLoss: {loss_text}",
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
    plt.savefig("/tmp/pnl.png")
    plt.close()
