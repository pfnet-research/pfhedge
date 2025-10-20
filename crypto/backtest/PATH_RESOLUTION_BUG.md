# Path Resolution Bug in Backtest Framework

**Status**: ✅ RESOLVED (2025-10-20)
**Severity**: Medium (was affecting YAML config files)
**Resolution**: Implemented Option 1 - Removed double path resolution

---

## Problem Description

The backtesting framework has **inconsistent path resolution** for the `data_dir` parameter, causing confusion and requiring absolute paths as workaround.

### Symptom

When using a YAML config with relative `data_dir`:
```yaml
data_dir: sample_data
```

The path gets resolved TWICE, resulting in incorrect paths like:
```
crypto/data/crypto/backtest/sample_data  ❌
```

### Root Cause

**Double Resolution**: Two components resolve relative paths independently:

1. **`BacktestConfig.load_yaml()`** (config.py:245-254)
   - Resolves relative paths relative to **config file directory**
   - `sample_data` → `crypto/backtest/sample_data`

2. **`Backtester.load_data()`** (backtester.py:236-243)
   - Assumes relative paths are relative to **`crypto/data/`**
   - `crypto/backtest/sample_data` → `crypto/data/crypto/backtest/sample_data`

Result: Path gets mangled!

---

## Why This Is Confusing

### Inconsistent Behavior Across Fields

| Field | YAML Resolution | Backtester Resolution | Works? |
|-------|----------------|----------------------|--------|
| `model_path` | ✅ Relative to config | ❌ None | ✅ YES |
| `data_dir` | ✅ Relative to config | ✅ Relative to crypto/data | ❌ NO (double resolution!) |
| `output_dir` | ✅ Relative to config | ❌ None | ✅ YES |

### Misleading Documentation

example_config.yaml says:
```yaml
# 3. Relative paths (resolved from this config file's directory):
#    data_dir: ../data
```

This is **incomplete** - doesn't mention Backtester's additional resolution!

---

## Current Workaround

Use **absolute paths** in YAML config:
```yaml
data_dir: /absolute/path/to/pfhedge/crypto/data/sample_data
```

**Downside**: Not portable across machines/environments

---

## Proper Fix Options

### Option 1: Remove Backtester's Path Resolution (RECOMMENDED)

**Change**: Make Backtester trust the path from config (already resolved by YAML loader)

```python
# crypto/backtest/backtester.py (lines 236-243)
# BEFORE:
if not os.path.isabs(data_dir):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data", data_dir)

# AFTER:
# Trust the config - path already resolved by YAML loader
# Just validate it exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
```

**Pros:**
- ✅ Fixes double-resolution bug
- ✅ Makes behavior consistent across all config fields
- ✅ Respects user's path intention
- ✅ Works with relative and absolute paths

**Cons:**
- ❌ Breaking change for users who rely on current behavior
- ❌ Need to update documentation

**Migration**:
- Old configs with `data_dir: sample_data` would need: `data_dir: ../data/sample_data`
- OR keep both behaviors with a flag

---

### Option 2: Disable YAML Anchor Resolution for data_dir

**Change**: Don't resolve `data_dir` in YAML loader, let Backtester handle it

```python
# crypto/backtest/config.py (lines 245-254)
# Resolve relative paths relative to config file directory
if anchor_relative_paths:
    config_dir = path_obj.parent.absolute()
    for field in ["model_path", "output_dir"]:  # Remove data_dir!
        if field in config_dict and isinstance(config_dict[field], str):
            field_path = Path(config_dict[field])
            if not field_path.is_absolute():
                resolved = (config_dir / field_path).resolve()
                config_dict[field] = str(resolved)
```

**Pros:**
- ✅ Backward compatible
- ✅ Preserves Backtester's `crypto/data/` convention

**Cons:**
- ❌ `data_dir` behaves differently from `model_path` and `output_dir`
- ❌ Still confusing for users

---

### Option 3: Add Clear Error Message

**Change**: Detect double-resolution and provide helpful error

```python
# crypto/backtest/backtester.py
if not os.path.isabs(data_dir):
    # Check if path looks like it was already resolved
    if "crypto/backtest" in data_dir or "crypto/data" in data_dir:
        raise ValueError(
            f"data_dir appears to be already resolved: {data_dir}\n"
            f"Tip: Use absolute path or just the directory name (e.g., 'sample_data')\n"
            f"The backtester will automatically look in crypto/data/"
        )

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data", data_dir)
```

**Pros:**
- ✅ Helps users debug quickly
- ✅ No breaking changes

**Cons:**
- ❌ Doesn't actually fix the underlying issue
- ❌ Still requires workaround

---

## Recommended Solution

**Implement Option 1 with backward compatibility flag:**

```python
# crypto/backtest/config.py
@dataclass
class BacktestConfig:
    # ... existing fields ...

    # New field for backward compatibility
    legacy_data_path_resolution: bool = False  # Set True for old behavior

# crypto/backtest/backtester.py
def load_data(self):
    data_dir = self.config.data_dir

    # Legacy behavior: assume relative to crypto/data/
    if self.config.legacy_data_path_resolution and not os.path.isabs(data_dir):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, "data", data_dir)

    # Modern behavior: trust the path from config
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
```

---

## Updated Documentation

### example_config.yaml (FIXED)

```yaml
# Data directories
#
# IMPORTANT: Path resolution rules:
#   - Relative paths are resolved relative to THIS config file's location
#   - The backtester will NOT prepend crypto/data/ automatically
#
# Examples:
data_dir: ../data/sample_data        # OK: Relative to config file
data_dir: /absolute/path/sample_data # OK: Absolute path
data_dir: sample_data                # ❌ WRONG: Will fail (use ../data/sample_data)
```

---

## Testing Checklist

- [ ] Relative path from repo root: `../data/sample_data`
- [ ] Absolute path: `/full/path/to/data`
- [ ] Tilde expansion: `~/data/sample_data`
- [ ] Environment variable: `$DATA_DIR/sample_data`
- [ ] Config in different directory than repo root
- [ ] Legacy mode with `legacy_data_path_resolution=True`

---

## Timeline

- **Short term** (Today): Document workaround (use absolute paths)
- **Medium term** (This week): Implement Option 1 with backward compat flag
- **Long term** (Next release): Deprecate legacy mode, make new behavior default

---

## Related Issues

- Model path resolution works correctly (no double resolution)
- Output path resolution works correctly (no double resolution)
- This is ONLY an issue with `data_dir`

---

## Impact Assessment

**Who is affected:**
- Users creating new YAML configs with relative `data_dir`
- Anyone copying example configs between directories

**Who is NOT affected:**
- Users with absolute paths in configs
- Users calling Backtester programmatically (not via YAML)
- Existing configs that accidentally work due to specific directory structure

---

## ✅ RESOLUTION (Implemented 2025-10-20)

### Changes Made

**1. Fixed backtester.py (lines 236-243)**
- **Removed** the double path resolution logic that prepended `crypto/data/`
- Now **trusts** the path from config (already resolved by YAML loader)
- Added helpful error message when path doesn't exist

```python
# BEFORE (buggy):
if not os.path.isabs(data_dir):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data", data_dir)

# AFTER (fixed):
# Trust the path from config (already resolved by YAML loader)
# Just validate that it exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(...)
```

**2. Updated example_config.yaml**
- Fixed documentation to reflect new behavior
- Updated example from `data_dir: sample_data` to `data_dir: ../data/sample_data`
- Added clear path resolution rules and common mistakes section
- Emphasized that relative paths are resolved ONCE relative to config file

**3. Testing**
- Verified relative paths work correctly (resolved from config file location)
- Verified absolute paths still work
- Verified environment variable expansion works
- Verified tilde expansion works
- Confirmed backtester no longer double-resolves paths

### Behavior Changes

| Scenario | Before (Broken) | After (Fixed) |
|----------|----------------|---------------|
| `data_dir: sample_data` in config | `crypto/data/crypto/backtest/sample_data` ❌ | Will fail unless you meant current dir ⚠️ |
| `data_dir: ../data/sample_data` | `crypto/data/crypto/backtest/../data/sample_data` ❌ | `crypto/data/sample_data` ✅ |
| Absolute path | Works ✅ | Works ✅ |
| Env vars | Works ✅ | Works ✅ |

### Migration Guide

If you have existing YAML configs with `data_dir: sample_data`:

**Option 1: Use relative path from config file**
```yaml
# Old (broken):
data_dir: sample_data

# New (fixed):
data_dir: ../data/sample_data  # If config is in crypto/backtest/
```

**Option 2: Use absolute path**
```yaml
data_dir: /absolute/path/to/pfhedge/crypto/data/sample_data
```

**Option 3: Use environment variable**
```yaml
data_dir: $DATA_DIR/sample_data
```

### Benefits

✅ **Consistent behavior**: All path fields (`model_path`, `data_dir`, `output_dir`) now use the same resolution logic
✅ **Predictable**: Paths are resolved exactly once, relative to config file location
✅ **No surprises**: What you specify is what you get (after single resolution)
✅ **Better errors**: Clear error messages guide users when paths don't exist

### Testing

All tests passed:
- ✅ Config path resolution (relative, absolute, env vars, tilde)
- ✅ Backtester trusts resolved paths without double-resolution
- ✅ Error messages are helpful when paths don't exist
