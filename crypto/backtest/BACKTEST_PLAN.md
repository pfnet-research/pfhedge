# Bitcoin Options Backtesting Framework - Implementation Plan

## Goal
Build a backtesting framework to evaluate pre-trained deep hedging models on historical Bitcoin data with real option price comparison.

## Requirements Summary

### Core Requirements
1. ✅ Load pre-trained model (assume checkpoint exists)
2. ✅ Download/load historical data for specified date range
3. ✅ Generate multiple bootstrap paths from historical data
4. ✅ Run two strategies: Deep Hedge vs Black-Scholes Delta
5. ✅ Compare model prices with real Deribit option prices
6. ✅ Generate comprehensive reports and plots

### Key Design Principles
- **Incremental development**: One small feature at a time
- **Test-driven**: Test each feature before moving on
- **Reuse existing code**: Leverage `deep_hedge_utils.py`, `BitcoinPerpetualHistorical`, etc.
- **Simple first, complex later**: MVP before advanced features

---

## Phase 1: Foundation (Baby Steps)

### Step 1.1: Create BacktestConfig ✅
**File:** `crypto/backtest/config.py`
**What:** Simple configuration dataclass
**Test:** Can create config, validate fields, convert to/from dict
**Status:** COMPLETED - All 18 tests passing

```python
@dataclass
class BacktestConfig:
    # Date range
    start_date: str
    end_date: str

    # Option params
    strike: float
    maturity_days: int
    call: bool

    # Model
    model_path: str

    # Execution
    n_bootstrap_paths: int = 100
    transaction_cost: float = 0.0005
    dt_hours: float = 8.0
```

**Review point:** Config created, validated, tested

---

### Step 1.2: Create Metrics Calculator ✅
**File:** `crypto/backtest/metrics.py`
**What:** Functions to calculate individual metrics
**Test:** Test each metric function independently
**Status:** COMPLETED - All 24 tests passing

```python
def calculate_sharpe_ratio(pnl: Tensor) -> float
def calculate_max_drawdown(pnl: Tensor) -> float
def calculate_cvar(pnl: Tensor, alpha: float = 0.05) -> float
def calculate_win_rate(pnl: Tensor) -> float
```

**Review point:** Each metric tested with known inputs/outputs

---

### Step 1.3: Create Simple Backtester Shell ✅
**File:** `crypto/backtest/backtester.py`
**What:** Class structure with stub methods
**Test:** Can instantiate, methods exist (even if empty)
**Status:** COMPLETED - All 8 tests passing

```python
class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def load_model(self) -> Hedger:
        """Load pre-trained model from checkpoint."""
        pass  # TODO

    def load_data(self) -> CryptoDataLoader:
        """Load historical data."""
        pass  # TODO

    def run(self) -> BacktestResults:
        """Run full backtest."""
        pass  # TODO
```

**Review point:** Structure approved, ready to fill in

---

### Step 1.4: Implement Model Loading ✅
**File:** `crypto/backtest/backtester.py` (update)
**What:** Implement `load_model()` method with hardening improvements
**Test:** Can load a saved model checkpoint
**Status:** COMPLETED - All 11 tests passing (5 original + 6 hardening tests)

**Improvements Applied:**
- ✅ Device parameter for CPU/GPU support
- ✅ Backward compatibility (accepts both 'criterion' and 'risk_measure')
- ✅ Safer loading with `weights_only=True` (with fallback)
- ✅ Strict state dict checking with clear error messages
- ✅ Default features fallback to DEFAULT_FEATURES

**Review point:** Model loading works with test checkpoint

---

### Step 1.5: Implement Data Loading ✅
**File:** `crypto/backtest/backtester.py` (update)
**What:** Implement `load_data()` method using existing `CryptoDataLoader`
**Test:** Can load sample historical data
**Status:** COMPLETED - All 5 tests passing

**Features Implemented:**
- ✅ Load perpetual and options data from parquet files
- ✅ Verify data directory exists
- ✅ Handle missing options data gracefully (warning only)
- ✅ Validate perpetual data is not empty
- ✅ Print comprehensive data summary (records, date ranges, price ranges, spreads)
- ✅ Verify real market data properties (timestamps, prices, etc.)

**Tests Added:**
1. `test_load_data_success` - Verifies successful data loading
2. `test_load_data_directory_not_found` - Error when directory missing
3. `test_load_data_no_perpetual_files` - Error when no data files
4. `test_load_data_missing_options_is_ok` - Options data optional
5. `test_load_data_verifies_real_market_data` - Validates data properties

**Review point:** Data loading works with real market data, comprehensive error handling

---

### Step 1.6: Create Bootstrap Path Generator ✅
**File:** `crypto/backtest/backtester.py` (update)
**What:** Method to create option with bootstrap paths
**Test:** Generates correct number of paths from historical data
**Status:** COMPLETED - All 5 tests passing

**Features Implemented:**
- ✅ Creates `BitcoinPerpetualHistorical` with loaded data
- ✅ Generates bootstrap paths via `simulate_bootstrap()`
- ✅ Creates `BitcoinEuropeanOption` on top of bootstrap underlier
- ✅ Stores option in `self.option` for later use
- ✅ Prints comprehensive option summary (type, strike, paths, payoffs, ITM ratio)
- ✅ Handles both call and put options
- ✅ Falls back to `self.data_loader` if no argument provided

**Tests Added:**
1. `test_create_bootstrap_option_success` - Basic successful creation
2. `test_create_bootstrap_option_uses_self_data_loader` - Uses stored data loader
3. `test_create_bootstrap_option_no_data_loaded` - Error without data
4. `test_create_bootstrap_option_correct_time_steps` - Verifies time step calculation
5. `test_create_bootstrap_option_put_option` - Creates put options correctly

**Review point:** Bootstrap generation working with multiple paths from historical data

---

### Step 1.7: Implement Deep Hedge Evaluation
**File:** `crypto/backtest/backtester.py` (update)
**What:** Run deep hedge strategy on bootstrap paths
**Test:** Produces PnL tensor with correct shape

```python
def run_deep_hedge(self, option, model) -> Tensor:
    """Run deep hedging strategy."""
```

**Review point:** Deep hedge runs and produces results

---

### Step 1.8: Implement BS Baseline Evaluation
**File:** `crypto/backtest/backtester.py` (update)
**What:** Run BS delta hedge (reuse `calculate_bs_hedge_pnl`)
**Test:** Produces PnL tensor matching deep hedge shape

```python
def run_bs_baseline(self, option) -> Tensor:
    """Run Black-Scholes delta hedge."""
```

**Review point:** BS baseline runs and produces results

---

### Step 1.9: Create Results Container (Simple)
**File:** `crypto/backtest/results.py`
**What:** Store results and calculate basic metrics
**Test:** Can store data, calculate mean/std/sharpe

```python
class BacktestResults:
    def __init__(self, deep_pnl, bs_pnl, spots, positions):
        self.deep_pnl = deep_pnl
        self.bs_pnl = bs_pnl
        # ...

    def summary(self) -> dict:
        """Calculate summary statistics."""

    def to_dict(self) -> dict:
        """Export to dictionary."""
```

**Review point:** Results container working

---

### Step 1.10: Connect Everything (MVP)
**File:** `crypto/backtest/backtester.py` (update `run()`)
**What:** Wire all pieces together
**Test:** End-to-end backtest on 1-day sample data

**Review point:** MVP backtesting working!

---

## Phase 2: Reporting & Visualization

### Step 2.1: Add Plotting - PnL Comparison
**File:** `crypto/backtest/results.py` (add method)
**What:** Plot cumulative PnL comparison
**Test:** Generates correct plot

### Step 2.2: Add Plotting - Distribution
**File:** `crypto/backtest/results.py` (add method)
**What:** Plot PnL distribution histograms
**Test:** Generates correct plot

### Step 2.3: Add Plotting - Hedge Positions
**File:** `crypto/backtest/results.py` (add method)
**What:** Plot hedge positions over time
**Test:** Generates correct plot

### Step 2.4: Add All Risk Metrics
**File:** `crypto/backtest/metrics.py` (expand)
**What:** Add CVaR, max drawdown, Sortino, VaR
**Test:** Each metric tested independently

### Step 2.5: Generate Report
**File:** `crypto/backtest/results.py` (add method)
**What:** Create markdown/HTML summary report
**Test:** Report generation works

---

## Phase 3: Real Option Price Comparison

### Step 3.1: Load Real Option Data
**File:** `crypto/backtest/option_comparison.py`
**What:** Load matching options from Deribit data
**Test:** Can find and load matching options

### Step 3.2: Calculate Model-Implied Prices
**File:** `crypto/backtest/option_comparison.py`
**What:** Calculate model's option value estimate
**Test:** Produces reasonable values

### Step 3.3: Price Comparison Analysis
**File:** `crypto/backtest/option_comparison.py`
**What:** Compare model vs market prices
**Test:** Generates comparison metrics and plots

### Step 3.4: IV Analysis
**File:** `crypto/backtest/option_comparison.py`
**What:** Compare implied volatilities
**Test:** Generates IV comparison plots

---

## Phase 4: Polish & CLI

### Step 4.1: Config File Support
**File:** `crypto/backtest/config.py` (update)
**What:** YAML load/save
**Test:** Can read/write YAML configs

### Step 4.2: CLI Interface
**File:** `crypto/backtest/run.py`
**What:** Command-line interface
**Test:** Can run backtest from command line

### Step 4.3: Documentation
**Files:** Docstrings, examples
**What:** Complete documentation
**Test:** Examples run successfully

---

## Current Status
- [ ] Phase 1: Foundation (Steps 1.1 - 1.10)
  - [x] Step 1.1: Create BacktestConfig
  - [x] Step 1.2: Create Metrics Calculator
  - [x] Step 1.3: Create Simple Backtester Shell
  - [x] Step 1.4: Implement Model Loading
  - [x] Step 1.5: Implement Data Loading
  - [x] Step 1.6: Create Bootstrap Path Generator
  - [ ] Step 1.7: Implement Deep Hedge Evaluation
  - [ ] Step 1.8: Implement BS Baseline Evaluation
  - [ ] Step 1.9: Create Results Container
  - [ ] Step 1.10: Connect Everything (MVP)
- [ ] Phase 2: Reporting & Visualization
- [ ] Phase 3: Real Option Price Comparison
- [ ] Phase 4: Polish & CLI

## Next Step
Step 1.7: Implement Deep Hedge Evaluation
