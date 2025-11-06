# Diagnostic & Utility Scripts

## GPU Diagnostics

### diagnose_gpu.py
**Purpose**: Comprehensive GPU diagnostics and training speed testing

**Use cases**:
- Verify CUDA is available and working
- Check GPU memory usage
- Test tensor operations on GPU
- Verify pfhedge instruments create data on GPU correctly
- Benchmark actual training speed

**Usage**:
```bash
# Basic diagnostics (quick, ~5 seconds)
python crypto/scripts/diagnose_gpu.py

# Full training speed test (1-2 minutes)
python crypto/scripts/diagnose_gpu.py --test-training

# Custom test with different parameters
python crypto/scripts/diagnose_gpu.py --test-training --n-paths 100000 --n-epochs 5
```

**When to use**:
- Training seems slow (>10s per iteration)
- Suspect GPU is not being used
- After changing GPU-related code
- Setting up new environment

## Performance Profiling

### profile_training.py
**Purpose**: Profile where time is spent during training

**Use cases**:
- Identify performance bottlenecks
- Understand which operations are slow
- Optimize training pipeline

**Usage**:
```bash
python crypto/scripts/profile_training.py
```

**When to use**:
- Training is slow and you need to find why
- Optimizing training performance
- Comparing different model architectures

## Model Verification

### verify_checkpoint.py
**Purpose**: Verify model checkpoint structure and parameters

**Use cases**:
- Check if checkpoint can be loaded
- Verify strike and other parameters match expected values
- Debug checkpoint loading issues

**Usage**:
```bash
# Basic verification
python crypto/scripts/verify_checkpoint.py <path/to/model.pth>

# Verify with expected strike
python crypto/scripts/verify_checkpoint.py <path/to/model.pth> --expected-strike 110000
```

**When to use**:
- Model fails to load
- Suspect checkpoint corruption
- Need to verify model parameters

## Data Pipeline Verification

### verify_tardis.py
**Purpose**: Verify TARDIS API connection and data fetching

**Use cases**:
- Test TARDIS API key and connection
- Verify instrument data can be fetched
- Test option book data download
- Debug data pipeline issues

**Usage**:
```bash
# Basic verification (uses default BTC options)
python crypto/scripts/verify_tardis.py

# Test specific currency
python crypto/scripts/verify_tardis.py --currency ETH

# Test specific date range
python crypto/scripts/verify_tardis.py --start-date 2025-01-01 --end-date 2025-01-31
```

**When to use**:
- Setting up TARDIS API for first time
- Data download fails
- Suspect API key issues
- Debugging instrument selection

## Quick Reference

| Script | Purpose | Speed | When to Use |
|--------|---------|-------|-------------|
| `diagnose_gpu.py` | GPU diagnostics | Fast (5s) | GPU not working, slow training |
| `diagnose_gpu.py --test-training` | Training speed test | Slow (1-2min) | Benchmark GPU performance |
| `profile_training.py` | Performance profiling | Medium (30s) | Find bottlenecks |
| `verify_checkpoint.py` | Model verification | Fast (<1s) | Checkpoint loading issues |
| `verify_tardis.py` | Data pipeline test | Medium (10-30s) | Data download issues |

## Common Workflows

### Debugging Slow Training
1. Run `diagnose_gpu.py` to check GPU is available
2. If GPU OK, run `diagnose_gpu.py --test-training` to benchmark
3. If still slow, run `profile_training.py` to find bottleneck

### Setting Up New Environment
1. Run `diagnose_gpu.py` to verify GPU works
2. Run `verify_tardis.py` to test data pipeline
3. Run small training job to verify end-to-end

### Debugging Model Loading
1. Run `verify_checkpoint.py <model.pth>` to check structure
2. Check strike and other parameters match
3. Verify checkpoint was saved correctly

## Expected Performance

### GPU Training (50k paths, 3 epochs)
- **GPU (working)**: ~3-6s per epoch
- **CPU bottleneck**: 10-15s per epoch
- **CPU only**: >20s per epoch

If you see >10s per epoch, GPU is likely not being used properly.

### GPU Memory Usage (100k paths, LSTM)
- **Typical usage**: 40-45GB
- **Model only**: ~100MB
- **Data (100k paths)**: ~40GB

If GPU memory is <1GB during training, data is likely on CPU.
