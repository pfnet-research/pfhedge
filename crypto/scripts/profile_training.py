#!/usr/bin/env python3
"""
Profile where time is being spent during training.
"""

import sys
import torch
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.training import TrainingConfig, Trainer


def profile_training():
    print("=" * 70)
    print("TRAINING PROFILER")
    print("=" * 70)

    config = TrainingConfig(
        strike=1.0062,
        maturity_days=30,
        model_path="/tmp/profile_test.pth",
        call=True,
        volatility=0.42,
        dt_hours=8.0,
        n_paths=10000,  # Smaller for quick profiling
        n_epochs=1,
        n_layers=2,
        n_units=128,
        risk_measure="entropic",
        risk_param=2.0,
        train_seed=42,
        device="cuda",
        underlying_type="spot",
        model_type="lstm",
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=1e-3,
        use_amp=True,
    )

    trainer = Trainer(config, verbose=False)

    # Time option creation
    print("\n1. Creating option...")
    start = time.time()
    option = trainer.create_option(n_paths=config.n_paths, seed=config.train_seed)
    t_option = time.time() - start
    print(f"   Time: {t_option:.3f}s")
    print(f"   Device: {option.underlier.spot.device}")
    print(f"   Shape: {option.underlier.spot.shape}")

    # Time model creation
    print("\n2. Creating model...")
    start = time.time()
    model = trainer.create_model()
    t_model = time.time() - start
    print(f"   Time: {t_model:.3f}s")

    # Time a single simulation
    print("\n3. Testing simulation speed (10 trials)...")
    sim_times = []
    for i in range(10):
        start = time.time()
        option.underlier.simulate(n_paths=config.n_paths, time_horizon=30 / 365)
        torch.cuda.synchronize()  # Wait for GPU to finish
        t_sim = time.time() - start
        sim_times.append(t_sim)
        if i == 0:
            print(f"   First simulation: {t_sim:.3f}s (includes compilation)")
        elif i == 9:
            print(f"   Last simulation: {t_sim:.3f}s")

    avg_sim = sum(sim_times[1:]) / len(sim_times[1:])  # Skip first (cold start)
    print(f"   Average (warm): {avg_sim:.3f}s")

    # Check if simulation is on GPU
    if option.underlier.spot.device.type == "cuda":
        print(f"   ✅ Simulation is on GPU")
    else:
        print(f"   ❌ Simulation is on CPU!")

    # Time a forward pass
    print("\n4. Testing forward pass speed...")
    start = time.time()
    with torch.no_grad():
        output = model.compute_pl(option)
    torch.cuda.synchronize()
    t_forward = time.time() - start
    print(f"   Time: {t_forward:.3f}s")

    # Time a full loss computation (simulation + forward + loss)
    print("\n5. Testing full loss computation...")
    start = time.time()
    loss = model.compute_loss(option, n_paths=config.n_paths)
    torch.cuda.synchronize()
    t_loss = time.time() - start
    print(f"   Time: {t_loss:.3f}s")
    print(f"   Loss value: {loss.item():.6f}")

    # Time a backward pass
    print("\n6. Testing backward pass...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    start = time.time()
    loss = model.compute_loss(option, n_paths=config.n_paths)
    loss.backward()
    torch.cuda.synchronize()
    t_backward = time.time() - start
    print(f"   Time: {t_backward:.3f}s")

    # Summary
    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)
    print(
        f"Simulation (avg):        {avg_sim:.3f}s  ({avg_sim/t_loss*100:.1f}% of loss computation)"
    )
    print(f"Full loss computation:   {t_loss:.3f}s")
    print(f"Backward pass:           {t_backward:.3f}s")
    print(f"\nEstimated time/epoch:    {t_backward:.3f}s")

    # Analysis
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)
    if avg_sim / t_loss > 0.5:
        print("⚠️  Simulation takes >50% of time")
        print("   This is normal - simulation generates random numbers on GPU")
    else:
        print("✅ Simulation is fast relative to model computation")

    if t_backward < 2.0:
        print(f"✅ Training is efficient ({t_backward:.2f}s per step)")
    elif t_backward < 5.0:
        print(f"⚠️  Training is moderate ({t_backward:.2f}s per step)")
    else:
        print(f"❌ Training is slow ({t_backward:.2f}s per step)")


if __name__ == "__main__":
    profile_training()
