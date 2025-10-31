# Deep Hedging Model Training - User Guide

Complete guide for training deep hedging models for Bitcoin options.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Running Training](#running-training)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)

---

## Quick Start

### 1. Run Training with Defaults

```bash
python -m crypto.training.train_model
```

That's it! The model will be trained and saved to `models/deep_hedger_trained.pth`.

### 2. Run with Custom Parameters

```bash
python -m crypto.training.train_model \
    --strike 60000 \
    --vol 1.0 \
    --epochs 100 \
    --paths 20000
```

### 3. View All Options

```bash
python -m crypto.training.train_model --help
```

---

## Configuration

### Command-Line Arguments

The training script supports extensive customization via CLI arguments.

#### Option Parameters

```bash
--strike 50000          # Option strike price (default: 50000)
--maturity 14           # Option maturity in days (default: 14)
--call                  # Call option (default: True)
--put                   # Put option (overrides --call)
```

#### Market Parameters

```bash
--vol 0.8               # Volatility (default: 0.8 = 80%)
--drift 0.0             # Drift for simulation (default: 0.0)
--cost 0.0005           # Transaction cost rate (default: 0.0005 = 0.05%)
--dt-hours 8.0          # Time step in hours (default: 8.0)
```

#### Training Parameters

```bash
--paths 10000           # Number of training paths (default: 10000)
--epochs 80             # Number of training epochs (default: 80)
--seed 42               # Random seed for reproducibility (default: 42)
```

#### Model Architecture

```bash
--layers 4              # Number of hidden layers (default: 4)
--units 128             # Number of units per layer (default: 128)
--risk-measure expected_shortfall  # Risk measure (default: expected_shortfall)
--risk-param 0.9        # Risk parameter, e.g., CVaR alpha (default: 0.9)
```

Available risk measures:
- `expected_shortfall` (CVaR)
- `variance`
- `cvar`
- `entropic`

#### Output Settings

```bash
--model-path models/my_model.pth    # Path to save trained model
--output-dir my_results             # Directory for training outputs
```

#### Test Parameters

```bash
--test-paths 200        # Number of test paths for evaluation (default: 200)
```

#### Device Selection

```bash
--device cpu            # Use CPU (default)
--device cuda           # Use first GPU
--device cuda:0         # Use specific GPU
```

### Python API

The Python API is **silent by default** (perfect for notebooks and scripts). You can enable verbose output if desired.

#### Silent Mode (Default)

```python
from crypto.training import TrainingConfig, Trainer

# Create config
config = TrainingConfig(
    strike=50000,
    maturity_days=14,
    call=True,
    volatility=0.8,
    transaction_cost=0.0005,
    n_paths=10000,
    n_epochs=80,
    model_path="models/deep_hedger.pth",
)

# Run training (silent - no output except progress bar)
trainer = Trainer(config, verbose=False)  # Default
results = trainer.train(seed=42)

# Analyze results
summary = results.summary()
print(f"Final Loss: {summary['final_loss']:.6f}")
print(f"Improvement: {summary['improvement_pct']:.1f}%")
```

#### Verbose Mode

```python
# Enable verbose output (like CLI)
trainer = Trainer(config, verbose=True)
results = trainer.train(seed=42)

# Prints full configuration banner, progress, and summary
```

---

## Running Training

### Method 1: CLI with Default Settings

Fastest way to get started:

```bash
python -m crypto.training.train_model
```

### Method 2: CLI with Custom Parameters

Train a put option with high volatility:

```bash
python -m crypto.training.train_model \
    --put \
    --strike 60000 \
    --vol 1.2 \
    --maturity 30 \
    --epochs 100 \
    --paths 20000 \
    --seed 42
```

### Method 3: Python Script

```python
from crypto.training import TrainingConfig, Trainer

config = TrainingConfig(
    strike=50000,
    maturity_days=14,
    call=True,
    volatility=0.8,
    n_paths=10000,
    n_epochs=80,
)

trainer = Trainer(config)
results = trainer.train(seed=42)

# Save results
results.to_json("training_results/results.json", include_raw=True)
```

### Method 4: Jupyter Notebook

See `quick_test.ipynb` sections 10-12 for interactive training examples.

---

## Understanding Results

### Console Output

During training, you'll see:

```
======================================================================
BITCOIN DEEP HEDGING - MODEL TRAINING
======================================================================

Configuration:
  Option: Call @ $50,000
  Maturity: 14 days
  Volatility: 80.0%
  Transaction cost: 0.05%
  Time step: 8.0 hours

Training:
  Paths: 10,000
  Epochs: 80
  Seed: 42

Model:
  Architecture: 4 layers × 128 units
  Risk measure: expected_shortfall (param=0.9)
  Device: cpu

Output:
  Model: models/deep_hedger_trained.pth
  Results: training_results/
======================================================================

Epoch 1/80 | Loss: 1234.5678 | Time: 2.3s
Epoch 2/80 | Loss: 987.6543 | Time: 2.1s
...
Epoch 80/80 | Loss: 123.4567 | Time: 2.0s

======================================================================
SUCCESS!
======================================================================

Final Training Loss: 123.456789
Improvement: 90.0%

Test Performance:
  Deep hedge: $1,234.56 ± $567.89 (Sharpe: 2.174)
  BS baseline: $987.65 ± $678.90 (Sharpe: 1.454)
  Improvement: $+246.91 (Sharpe: +0.720)

Model saved to: models/deep_hedger_trained.pth
======================================================================
```

### Generated Files

After training completes:

```
training_results/
└── training_results.json    # Complete training metrics and config

models/
└── deep_hedger_trained.pth  # Trained model checkpoint
```

### Metrics Explained

**Training Metrics:**
- **Loss**: Risk-adjusted loss value (lower is better)
- **Improvement**: Percentage reduction from initial to final loss
- **Time per Epoch**: Training speed indicator

**Test Performance:**
- **Mean PnL**: Average profit/loss on held-out test paths
- **Std PnL**: Standard deviation of PnL (risk measure)
- **Sharpe Ratio**: Risk-adjusted return (Mean / Std)

**Comparison:**
- **Improvement**: How much better deep hedge performs vs Black-Scholes
- **Sharpe Improvement**: Additional risk-adjusted return

### Results JSON Structure

```json
{
  "config": {
    "strike": 50000,
    "maturity_days": 14,
    "volatility": 0.8,
    ...
  },
  "training": {
    "initial_loss": 1234.5678,
    "final_loss": 123.4567,
    "improvement_pct": 90.0,
    "n_epochs": 80,
    "losses": [1234.56, 987.65, ...]
  },
  "test_metrics": {
    "deep_hedge": {
      "mean_pnl": 1234.56,
      "std_pnl": 567.89,
      "sharpe_ratio": 2.174,
      ...
    },
    "bs_baseline": {
      "mean_pnl": 987.65,
      "std_pnl": 678.90,
      "sharpe_ratio": 1.454,
      ...
    },
    "comparison": {
      "mean_pnl_improvement": 246.91,
      "sharpe_improvement": 0.720,
      ...
    }
  },
  "model_path": "models/deep_hedger_trained.pth",
  "timestamp": "2025-10-21T08:10:00"
}
```

---

## Advanced Usage

### Silent vs Verbose Training

The Trainer supports two modes:

**Silent Mode (Default)** - Ideal for notebooks and batch processing:
```python
# No output except progress bar
trainer = Trainer(config, verbose=False)
results = trainer.train(seed=42)
```

**Verbose Mode** - Full output like CLI:
```python
# Prints configuration, progress, and summary
trainer = Trainer(config, verbose=True)
results = trainer.train(seed=42)
```

### Training Multiple Models

```python
from crypto.training import TrainingConfig, Trainer

# Parameter grid
strikes = [45000, 50000, 55000]
volatilities = [0.6, 0.8, 1.0]

for strike in strikes:
    for vol in volatilities:
        config = TrainingConfig(
            strike=strike,
            volatility=vol,
            model_path=f"models/model_s{strike}_v{vol}.pth",
            output_dir=f"results/s{strike}_v{vol}",
        )

        # Use silent mode for batch processing
        trainer = Trainer(config, verbose=False)
        results = trainer.train(seed=42)

        summary = results.summary()
        print(f"Strike {strike}, Vol {vol}: "
              f"Sharpe = {summary['test_metrics']['deep_hedge']['sharpe_ratio']:.3f}")
```

### Custom Risk Measures

Train with different risk measures:

```bash
# CVaR (Expected Shortfall)
python -m crypto.training.train_model \
    --risk-measure expected_shortfall \
    --risk-param 0.9

# Variance
python -m crypto.training.train_model \
    --risk-measure variance

# Entropic risk
python -m crypto.training.train_model \
    --risk-measure entropic \
    --risk-param 0.1
```

### GPU Training

If you have CUDA available:

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Train on GPU
python -m crypto.training.train_model --device cuda --paths 50000 --epochs 200
```

### Analyzing Training Progress

```python
import json
import matplotlib.pyplot as plt

# Load results
with open("training_results/training_results.json") as f:
    data = json.load(f)

# Plot training curve
losses = data["training"]["losses"]
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.grid(True, alpha=0.3)
plt.savefig("training_curve.png", dpi=300)
```

### Resuming Training

The framework doesn't currently support resuming, but you can train incrementally:

```python
from crypto.training import Trainer
import torch

# Load existing model
model = torch.load("models/checkpoint.pth")

# Create new trainer with more epochs
config = TrainingConfig(n_epochs=50)  # Additional epochs
trainer = Trainer(config)

# Initialize with pre-trained weights
trainer.hedger.load_state_dict(model.state_dict())

# Continue training
results = trainer.train(seed=43)
```

---

## API Reference

### Core Classes

#### `TrainingConfig`

Configuration dataclass for model training.

**Parameters:**
- `strike` (float): Option strike price
- `maturity_days` (int): Option maturity in days
- `call` (bool): True for call, False for put
- `volatility` (float): Volatility for simulation (e.g., 0.8 for 80%)
- `drift` (float): Drift for simulation
- `transaction_cost` (float): Transaction cost rate
- `dt_hours` (float): Time step in hours
- `n_paths` (int): Number of training paths
- `n_epochs` (int): Number of training epochs
- `n_layers` (int): Number of hidden layers
- `n_units` (int): Number of units per layer
- `risk_measure` (str): Risk measure for training
- `risk_param` (float): Risk parameter
- `model_path` (str): Path to save trained model
- `output_dir` (str): Directory for training outputs
- `test_n_paths` (int): Number of test paths
- `test_seed` (int): Random seed for test set
- `train_seed` (int): Random seed for training
- `device` (str): Device for training

**Methods:**
- `validate()`: Validate all parameters
- `to_dict()`: Convert to dictionary

#### `Trainer`

Main training engine.

**Constructor:**
- `__init__(config, verbose=False)`: Initialize trainer with config
  - `config` (TrainingConfig): Training configuration
  - `verbose` (bool): If True, print progress messages. Default: False (silent)
  - Raises `ValueError` if CUDA device requested but not available

**Methods:**
- `train(seed)`: Run complete training pipeline
  - Returns: `TrainingResults` object
- `create_option(n_paths, seed)`: Create option with simulated paths
  - Used for both training and testing with different parameters
- `create_model()`: Create deep hedger model
- `train_model()`: Train the model
- `evaluate()`: Evaluate model on test set
- `save_model()`: Save model checkpoint

**Usage:**
```python
# Silent mode (default - for notebooks)
trainer = Trainer(config)

# Verbose mode (for scripts/debugging)
trainer = Trainer(config, verbose=True)
```

#### `TrainingResults`

Container for training results and metrics.

**Properties:**
- `config`: Configuration used
- `initial_loss`: Loss at epoch 0
- `final_loss`: Loss at final epoch
- `losses`: List of losses per epoch
- `test_metrics`: Test performance metrics
- `model_path`: Path to saved model
- `timestamp`: Training completion time

**Methods:**
- `summary()`: Get comprehensive summary
- `to_dict()`: Export to dictionary
- `to_json(path, include_raw=False, indent=2)`: Save to JSON file

---

## Troubleshooting

### Common Issues

**1. "CUDA device requested but CUDA is not available"**

This error occurs when creating the Trainer with a CUDA device on a system without GPU support.

Solutions:
- Use `--device cpu` for CPU training (CLI)
- Use `device="cpu"` in config (Python API)
- Install PyTorch with CUDA support: https://pytorch.org/
- Check GPU availability: `nvidia-smi`

Note: The error is raised when creating `Trainer(config)`, not during `config.validate()`. This allows testing configs without requiring actual hardware.

**2. Training is slow**
- Reduce `--paths` (try 5000 or 2000)
- Reduce `--epochs` (try 50 or 30)
- Use GPU with `--device cuda`

**3. "Out of memory"**
- Reduce `--paths`
- Reduce `--units` (try 64 or 32)
- Reduce `--layers` (try 3 or 2)

**4. Poor test performance**
- Increase `--epochs` (try 150 or 200)
- Increase `--paths` (try 20000)
- Adjust `--risk-param` (try different values)
- Check if parameters match use case

**5. "Model file not found" when using trained model**
- Check `--model-path` value
- Verify file was created: `ls models/`

### Getting Help

1. Check examples in `quick_test.ipynb` (sections 10-12)
2. Review test files in `crypto/tests/`
3. Read docstrings: `help(Trainer.train)`

---

## Best Practices

1. **Always set `--seed`** for reproducibility
2. **Start small**: Test with 1000 paths and 10 epochs first
3. **Monitor convergence**: Check if loss is still decreasing
4. **Save results**: Export JSON for each experiment
5. **Test on realistic scenarios**: Match training params to backtest params
6. **Use GPU when available**: Can speed up training 10-50x
7. **Validate on test set**: Always check test performance, not just training loss
8. **Version control**: Track configs and results

---

## Examples

### Example 1: Quick Training Run

```bash
python -m crypto.training.train_model --epochs 50 --paths 5000
```

### Example 2: High-Quality Model

```bash
python -m crypto.training.train_model \
    --strike 50000 \
    --vol 0.8 \
    --epochs 150 \
    --paths 20000 \
    --layers 5 \
    --units 256 \
    --seed 42 \
    --model-path models/high_quality_hedger.pth
```

### Example 3: Put Option Model

```bash
python -m crypto.training.train_model \
    --put \
    --strike 45000 \
    --maturity 30 \
    --vol 1.0 \
    --cost 0.001 \
    --epochs 100 \
    --seed 123
```

### Example 4: Batch Training Script

```python
#!/usr/bin/env python3
import subprocess

configs = [
    {"strike": 45000, "vol": 0.6},
    {"strike": 50000, "vol": 0.8},
    {"strike": 55000, "vol": 1.0},
]

for i, cfg in enumerate(configs):
    print(f"\n{'='*70}")
    print(f"Training model {i+1}/{len(configs)}")
    print(f"{'='*70}\n")

    subprocess.run([
        "python", "-m", "crypto.training.train_model",
        "--strike", str(cfg["strike"]),
        "--vol", str(cfg["vol"]),
        "--epochs", "80",
        "--paths", "10000",
        "--model-path", f"models/model_{i+1}.pth",
        "--output-dir", f"results/run_{i+1}",
        "--seed", "42",
    ])
```

---

## See Also

- `crypto/training/trainer.py` - Core training implementation
- `crypto/training/config.py` - Configuration dataclass
- `crypto/training/results.py` - Results handling
- `quick_test.ipynb` - Interactive training examples (sections 10-12)
- `crypto/backtest/USAGE.md` - Backtesting guide

---

*Last updated: 2025-10-21*
*Framework version: Phase 4 Complete*
*Note: Python API is silent by default (v2.0+). Use `verbose=True` for detailed output.*
