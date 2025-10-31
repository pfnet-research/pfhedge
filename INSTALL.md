# PFHedge Installation Guide

Complete installation instructions for the PFHedge deep hedging framework with crypto extensions.

---

## Quick Start (Recommended)

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/your-repo/pfhedge.git
cd pfhedge

# 2. Install in development mode with all dependencies
pip install -e ".[backtest,crypto,dev]"

# 3. Verify installation
python -c "import pfhedge; import crypto; print('✅ Installation successful!')"
```

---

## Installation Methods

### Method 1: Editable Install (Recommended for Development)

Install the package in editable mode so code changes take effect immediately:

```bash
# Install core dependencies only
pip install -e .

# OR: Install with specific extras
pip install -e ".[backtest,crypto]"

# OR: Install everything (including dev tools)
pip install -e ".[backtest,crypto,dev]"
```

### Method 2: Standard Install

```bash
# Install core only
pip install .

# With extras
pip install ".[backtest,crypto]"
```

### Method 3: From GitHub (for users)

```bash
pip install git+https://github.com/your-repo/pfhedge.git
```

---

## Dependency Groups

### Core Dependencies (Always Installed)

```
torch>=1.9.0,<3.0.0       # PyTorch for deep learning
tqdm>=4.62.3               # Progress bars
numpy>=1.26                # Numerical computing
tardis-client>=2.0.0       # Crypto market data
requests>=2.25.0           # HTTP requests
```

### Optional: Backtest Dependencies

For running backtests and generating reports:

```bash
pip install ".[backtest]"
```

Includes:
- matplotlib>=3.3.0 (plotting)
- pandas>=1.3.0 (data analysis)
- scipy>=1.7.0 (statistics)
- pyyaml>=5.4.0 (config files)

### Optional: Crypto Dependencies

For crypto option trading and deep hedging:

```bash
pip install ".[crypto]"
```

Includes:
- pandas>=1.3.0
- pyyaml>=5.4.0

### Optional: Development Dependencies

For contributing to the project:

```bash
pip install ".[dev]"
```

Includes testing, linting, formatting, and documentation tools.

---

## Python Version Requirements

- **Minimum**: Python 3.8.1
- **Recommended**: Python 3.9+ (for better numpy compatibility)
- **Tested**: Python 3.9, 3.10, 3.11

---

## GPU Support (Optional)

### For NVIDIA GPUs (CUDA)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If not available, install PyTorch with CUDA support:
# Visit https://pytorch.org/get-started/locally/ and follow instructions

# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For Apple Silicon (MPS)

PyTorch automatically uses Metal Performance Shaders (MPS) on M1/M2/M3 Macs:

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## Verification

### Test Core Functionality

```bash
# Test pfhedge
python -c "import pfhedge; print(pfhedge.__version__)"

# Test crypto module
python -c "import crypto; print('Crypto module loaded')"

# Test GPU (if applicable)
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Run Tests

```bash
# Install test dependencies
pip install ".[dev]"

# Run all tests
pytest

# Run specific tests
pytest crypto/tests/

# Run with coverage
pytest --cov=pfhedge --cov=crypto
```

---

## Common Installation Issues

### Issue 1: PyTorch Installation Fails

**Problem**: `pip install torch` fails or installs CPU-only version

**Solution**:
```bash
# Use PyTorch's official installation command
# Visit: https://pytorch.org/get-started/locally/

# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Numpy Version Conflicts

**Problem**: `ImportError: numpy.core.multiarray failed to import`

**Solution**:
```bash
# Upgrade numpy
pip install --upgrade numpy

# OR: Install specific version for Python 3.8
pip install "numpy<1.25"
```

### Issue 3: Tardis Client Issues

**Problem**: `ModuleNotFoundError: No module named 'tardis_client'`

**Solution**:
```bash
pip install tardis-client>=2.0.0
```

### Issue 4: Permission Errors

**Problem**: `ERROR: Could not install packages due to an EnvironmentError`

**Solution**:
```bash
# Install for current user only
pip install --user -e ".[backtest,crypto]"

# OR: Use a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[backtest,crypto]"
```

---

## Virtual Environment Setup (Recommended)

### Using venv

```bash
# Create virtual environment
python -m venv pfhedge-env

# Activate (Linux/Mac)
source pfhedge-env/bin/activate

# Activate (Windows)
pfhedge-env\Scripts\activate

# Install dependencies
pip install -e ".[backtest,crypto,dev]"

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create environment
conda create -n pfhedge python=3.10

# Activate
conda activate pfhedge

# Install PyTorch (conda recommended for GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install pfhedge
pip install -e ".[backtest,crypto]"
```

---

## Quick Validation Script

Save this as `test_installation.py` and run it:

```python
#!/usr/bin/env python
"""Test PFHedge installation."""

def test_installation():
    print("="*60)
    print("TESTING PFHEDGE INSTALLATION")
    print("="*60)

    # Test imports
    try:
        import pfhedge
        print(f"✅ pfhedge {pfhedge.__version__}")
    except ImportError as e:
        print(f"❌ pfhedge: {e}")
        return False

    try:
        import crypto
        print("✅ crypto module")
    except ImportError as e:
        print(f"❌ crypto module: {e}")
        return False

    # Test PyTorch
    try:
        import torch
        print(f"✅ torch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False

    # Test optional dependencies
    optional = {
        'pandas': 'Data analysis',
        'matplotlib': 'Plotting',
        'scipy': 'Scientific computing',
        'yaml': 'Config files',
        'tardis_client': 'Market data',
    }

    print("\nOptional dependencies:")
    for module, desc in optional.items():
        try:
            __import__(module)
            print(f"  ✅ {module:15s} ({desc})")
        except ImportError:
            print(f"  ⚠️  {module:15s} ({desc}) - not installed")

    print("\n" + "="*60)
    print("INSTALLATION TEST COMPLETE")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_installation()
    exit(0 if success else 1)
```

Run it:
```bash
python test_installation.py
```

---

## Next Steps

After installation:

1. **Explore examples**: See `crypto/docs/STEP_BY_STEP_WORKFLOW.md`
2. **Train a model**: Try `crypto/scripts/train_for_option.py`
3. **Run backtest**: Use `python -m crypto.backtest.run`
4. **Read documentation**: Check `docs/` folder

---

## Getting Help

- **Issues**: https://github.com/your-repo/pfhedge/issues
- **Documentation**: See `crypto/docs/` for guides
- **Examples**: Check `crypto/examples/` for sample code

---

## License

MIT License - see LICENSE file for details
