#!/usr/bin/env python3

import argparse
import torch
import sys
from pathlib import Path
import time
import psutil
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def print_section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def diagnose_gpu():
    print_section("GPU DIAGNOSTICS")

    # 1. Basic CUDA info
    print("\n1. CUDA Availability:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("\n❌ CUDA is not available. Possible issues:")
        print("   - CUDA toolkit not installed")
        print("   - PyTorch installed without CUDA support")
        print("   - No GPU available on this machine")
        print(
            "\n   To fix: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )
        return

    # 2. GPU Details
    print(f"\n2. GPU Details:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"   - Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"   - CUDA Capability: {props.major}.{props.minor}")

    # 3. Current Memory Usage
    print(f"\n3. Current GPU Memory:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"   GPU {i}:")
        print(f"   - Allocated: {allocated:.3f} GB")
        print(f"   - Reserved:  {reserved:.3f} GB")

    # 4. Test tensor operations
    print(f"\n4. Testing Tensor Operations:")
    try:
        # Create a tensor on GPU
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000, device=device)
        print(f"   ✅ Created 1000x1000 tensor on {device}")

        # Perform computation
        start = time.time()
        result = torch.matmul(test_tensor, test_tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"   ✅ Matrix multiply completed in {elapsed:.4f}s")

        # Check memory after operation
        allocated = torch.cuda.memory_allocated(0) / 1e6
        print(f"   ✅ Memory after operation: {allocated:.1f} MB")

        # Cleanup
        del test_tensor, result
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"   ❌ Error during tensor operations: {e}")

    # 5. Check pfhedge specific issues
    print(f"\n5. Testing pfhedge GPU usage:")
    try:
        from crypto.instruments import BitcoinSpotBrownian
        from crypto.instruments import BitcoinEuropeanOption

        # Create underlier on GPU
        device = torch.device("cuda:0")
        underlier = BitcoinSpotBrownian(sigma=0.4, device=device)

        # Check device
        print(f"   Underlier device: {underlier.device}")
        if underlier.device.type == "cuda":
            print(f"   ✅ Underlier created on GPU")
        else:
            print(f"   ❌ Underlier still on CPU!")

        # Simulate and check
        underlier.simulate(n_paths=1000, time_horizon=30 / 365)
        if hasattr(underlier, "spot"):
            spot_device = underlier.spot.device
            spot_shape = underlier.spot.shape
            print(f"   Spot data device: {spot_device}")
            print(f"   Spot data shape: {spot_shape}")
            if spot_device.type == "cuda":
                print(f"   ✅ Data generated on GPU")
            else:
                print(f"   ❌ Data generated on CPU!")

    except Exception as e:
        print(f"   ❌ Error testing pfhedge: {e}")
        import traceback

        traceback.print_exc()

    # 6. System resources
    print(f"\n6. System Resources:")
    print(f"   CPU Usage: {psutil.cpu_percent()}%")
    print(f"   RAM Usage: {psutil.virtual_memory().percent}%")

    # 7. Check nvidia-smi
    print(f"\n7. nvidia-smi output:")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        lines = result.stdout.split("\n")
        # Find and print the process table
        printing = False
        for line in lines:
            if "Processes:" in line:
                printing = True
            if printing:
                print(f"   {line}")
    except Exception as e:
        print(f"   Could not run nvidia-smi: {e}")

    # Summary
    print_section("DIAGNOSIS SUMMARY")

    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print("✅ PyTorch can access GPU")

        # Test if our fix would work
        try:
            from crypto.instruments import BitcoinSpotBrownian

            device = torch.device("cuda:0")
            test = BitcoinSpotBrownian(device=device)
            test.simulate(100, 0.1)
            if test.spot.device.type == "cuda":
                print("✅ Fix should work: Options can be created on GPU")
            else:
                print("❌ Issue: Options still being created on CPU")
        except:
            print("⚠️  Could not test option creation")
    else:
        print("❌ No GPU access - training will use CPU")

    print("\nTo monitor during training:")
    print("  1. Run: watch -n 1 nvidia-smi")
    print("  2. Look for python process in the GPU process list")
    print("  3. Check iteration speed in training output (should be <2s/it)")


def test_training_speed(n_paths=50000, n_epochs=3):
    from crypto.training import TrainingConfig, Trainer

    print_section("TRAINING SPEED TEST")

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        initial_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"Initial GPU memory: {initial_memory:.2f} GB")

    config = TrainingConfig(
        strike=1.0062,
        maturity_days=30,
        model_path="/tmp/speed_test.pth",
        call=True,
        volatility=0.42,
        dt_hours=8.0,
        n_paths=n_paths,
        n_epochs=n_epochs,
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

    print(f"\nTest config: {config.n_paths:,} paths, {config.n_epochs} epochs")

    trainer = Trainer(config, verbose=True)

    print("\nCreating option...")
    start = time.time()
    option = trainer.create_option(n_paths=config.n_paths, seed=config.train_seed)
    option_time = time.time() - start

    if hasattr(option.underlier, "spot"):
        device = option.underlier.spot.device
        shape = option.underlier.spot.shape
        print(f"✅ Option created in {option_time:.2f}s")
        print(f"   Device: {device}")
        print(f"   Shape: {shape}")

        if torch.cuda.is_available():
            memory_after_option = torch.cuda.memory_allocated(0) / 1e9
            print(f"   GPU memory: {memory_after_option:.2f} GB")

    print("\nCreating model...")
    model = trainer.create_model()

    if torch.cuda.is_available():
        memory_after_model = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory: {memory_after_model:.2f} GB")

    print(f"\nTraining {n_epochs} epochs...")
    start_train = time.time()
    history = trainer.train_model(model, option)
    total_train_time = time.time() - start_train

    avg_time_per_epoch = total_train_time / config.n_epochs

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total training time: {total_train_time:.1f}s")
    print(f"Time per epoch: {avg_time_per_epoch:.2f}s")
    print(f"Loss history: {[f'{x:.6f}' for x in history]}")

    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated(0) / 1e9
        peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\nGPU Memory:")
        print(f"   Final: {final_memory:.2f} GB")
        print(f"   Peak: {peak_memory:.2f} GB")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if avg_time_per_epoch < 3.0:
        print("✅ EXCELLENT: Training speed indicates GPU is being used!")
        print(f"   {avg_time_per_epoch:.2f}s per epoch is fast")
    elif avg_time_per_epoch < 10.0:
        print("⚠️  MODERATE: Speed is okay but could be better")
        print(f"   {avg_time_per_epoch:.2f}s per epoch")
    else:
        print("❌ SLOW: Training is likely on CPU or has bottleneck")
        print(f"   {avg_time_per_epoch:.2f}s per epoch is too slow for GPU")

    print("\nFor reference (50k paths):")
    print("   - GPU training: < 3s per epoch")
    print("   - CPU bottleneck: 10-15s per epoch")
    print("   - Pure CPU: > 20s per epoch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose GPU usage and training performance"
    )
    parser.add_argument(
        "--test-training",
        action="store_true",
        help="Run actual training speed test (takes ~1-2 min)",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=50000,
        help="Number of paths for training test (default: 50000)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3,
        help="Number of epochs for training test (default: 3)",
    )

    args = parser.parse_args()

    if args.test_training:
        test_training_speed(n_paths=args.n_paths, n_epochs=args.n_epochs)
    else:
        diagnose_gpu()
