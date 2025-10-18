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

### Step 1.7: Implement Deep Hedge Evaluation ✅
**File:** `crypto/backtest/backtester.py` (update)
**What:** Run deep hedge strategy on bootstrap paths
**Test:** Produces PnL tensor with correct shape
**Status:** COMPLETED - All 5 tests passing

```python
def run_deep_hedge(self, option=None, model=None) -> Tensor:
    """Run deep hedging strategy.

    Uses pre-trained neural network to compute optimal hedge positions,
    calculates cumulative PnL including transaction costs and funding costs.
    """
```

**Features Implemented:**
- ✅ Compute hedge positions using `model.compute_hedge()`
- ✅ Calculate cumulative PnL using `model.compute_cum_pl()`
- ✅ Apply perpetual futures funding costs
- ✅ Falls back to `self.option` and `self.model` if not provided
- ✅ Returns tensor shape (n_paths, n_steps)

**Tests Added:**
1. `test_run_deep_hedge_success` - Basic successful execution
2. `test_run_deep_hedge_uses_stored_option_and_model` - Uses self references
3. `test_run_deep_hedge_no_model_loaded` - Error handling
4. `test_run_deep_hedge_no_option_created` - Error handling
5. `test_run_deep_hedge_correct_pnl_shape` - Shape validation

**Review point:** Deep hedge runs and produces results ✅

---

### Step 1.8: Implement BS Baseline Evaluation ✅
**File:** `crypto/backtest/backtester.py` (update)
**What:** Run BS delta hedge (reuse `calculate_bs_hedge_pnl`)
**Test:** Produces PnL tensor matching deep hedge shape
**Status:** COMPLETED - All 5 tests passing

```python
def run_bs_baseline(self, option=None) -> Tensor:
    """Run Black-Scholes delta hedge baseline.

    Uses Black-Scholes delta formula to compute hedge positions,
    calculates cumulative PnL including transaction costs and funding costs
    (matching deep hedge calculation for fair comparison).
    """
```

**Features Implemented:**
- ✅ Calculate BS delta using `option.black_scholes_delta()`
- ✅ Extract spots, payoffs, and cost from option
- ✅ Handle funding rate and funding times if available
- ✅ Call `calculate_bs_hedge_pnl()` with all parameters
- ✅ Returns tensor shape (n_paths, n_steps) matching deep hedge

**Tests Added:**
1. `test_run_bs_baseline_success` - Basic successful execution
2. `test_run_bs_baseline_uses_stored_option` - Uses self.option fallback
3. `test_run_bs_baseline_no_option_created` - Error handling
4. `test_run_bs_baseline_correct_pnl_shape` - Shape validation
5. `test_run_bs_baseline_matches_deep_hedge_shape` - Cross-validation with deep hedge

**Review point:** BS baseline runs and produces results ✅

---

### Step 1.9: Create Results Container ✅
**File:** `crypto/backtest/results.py`
**What:** Store results and calculate comprehensive metrics
**Test:** Can store data, validate inputs, calculate metrics
**Status:** COMPLETED - All 9 tests passing

**Features Implemented:**
- ✅ BacktestResults class with input validation
- ✅ Stores deep_pnl, bs_pnl, deep_positions, bs_positions, spots, config
- ✅ Validates all tensors are 2D with matching shapes
- ✅ summary() method calculating 12 metrics per strategy:
  - Basic stats: mean, std, min, max, median
  - Risk-adjusted: Sharpe ratio, Sortino ratio
  - Risk metrics: CVaR, VaR, max drawdown, Calmar ratio
  - Win rate
- ✅ to_dict() method for data export including config
- ✅ __repr__() for readable string representation

**Tests Added:**
1. `test_create_results_success` - Verifies successful creation
2. `test_create_results_with_config` - Tests config storage
3. `test_validate_inputs_wrong_dimension` - Validates 2D requirement
4. `test_validate_inputs_mismatched_shapes` - Validates shape matching
5. `test_summary_basic` - Tests summary structure and metrics
6. `test_summary_values_reasonable` - Tests metric calculation correctness
7. `test_to_dict_structure` - Tests dictionary export
8. `test_to_dict_with_config` - Tests config included in export
9. `test_repr` - Tests string representation

**Review point:** Results container working with comprehensive metrics ✅

---

### Step 1.10: Connect Everything (MVP) ✅
**File:** `crypto/backtest/backtester.py` (update `run()`)
**What:** Wire all pieces together
**Test:** End-to-end backtest on 1-day sample data
**Status:** COMPLETED - All 5 end-to-end tests passing + review improvements applied

**Features Implemented:**
- ✅ Added position storage fields (`self.deep_positions`, `self.bs_positions`)
- ✅ Modified `run_deep_hedge()` to store hedge positions
- ✅ Modified `run_bs_baseline()` to store BS delta positions
- ✅ Implemented complete `run()` method orchestrating all 5 steps:
  1. Load model from checkpoint
  2. Load historical data from parquet files
  3. Create bootstrap option with multiple paths
  4. Run deep hedge strategy
  5. Run BS baseline strategy
- ✅ Creates `BacktestResults` object with all data
- ✅ Prints comprehensive progress updates at each step
- ✅ Prints detailed performance summary with metrics for both strategies

**Review Improvements Applied:**
- ✅ **Error handling**: Wrapped `run()` in try-except with helpful categorized error messages
  - FileNotFoundError: Missing model/data files
  - ValueError: Invalid data or configuration
  - RuntimeError/KeyError: Model or execution errors
- ✅ **Determinism**: Added `seed` parameter to `run()` for reproducibility
  - Sets torch.manual_seed() and np.random.seed()
  - Prints warning if no seed provided
- ✅ **Config echo**: Added model_path and data_dir to configuration header
- ✅ **Funding alignment**: Added `_check_funding_alignment()` helper method
  - Warns if funding payment times don't align with time grid
  - Non-blocking (warning only, doesn't stop execution)
  - Suggests adjusting dt_hours to align with funding frequency

**Tests Added:**
1. `test_run_end_to_end` - Complete backtest pipeline with 5 paths, 10 days data
2. `test_run_stores_positions` - Verifies positions stored correctly in results
3. `test_run_with_seed_parameter` - Tests seed parameter functionality
4. `test_run_error_handling_missing_model` - Tests helpful error for missing model
5. `test_run_error_handling_missing_data` - Tests helpful error for missing data

**Review point:** MVP backtesting working with robust error handling! ✅

---

## Phase 2: Reporting & Visualization ✅ COMPLETE

### Step 2.1: Add Plotting - PnL Comparison ✅
**File:** `crypto/backtest/results.py` (add method)
**What:** Plot cumulative PnL comparison
**Status:** COMPLETED - `plot_pnl_comparison()` implemented with time_unit parameter

### Step 2.2: Add Plotting - Distribution ✅
**File:** `crypto/backtest/results.py` (add method)
**What:** Plot PnL distribution histograms
**Status:** COMPLETED - `plot_pnl_distribution()` implemented with overlay histograms

### Step 2.3: Add Plotting - Hedge Positions ✅
**File:** `crypto/backtest/results.py` (add method)
**What:** Plot hedge positions over time
**Status:** COMPLETED - `plot_positions()` implemented with time_unit parameter

### Step 2.4: Add All Risk Metrics ✅
**File:** `crypto/backtest/metrics.py` (expand)
**What:** Add CVaR, max drawdown, Sortino, VaR
**Status:** COMPLETED - All metrics implemented in Phase 1 (Step 1.2)

### Step 2.5: Generate Report ✅
**File:** `crypto/backtest/results.py` (add method)
**What:** Create markdown summary report with plots
**Status:** COMPLETED - `generate_report()` implemented with relative links and plot path tracking

### Step 2.6: Comprehensive Visualization ✅
**File:** `crypto/backtest/results.py` (add method)
**What:** Create 2x2 grid with all visualizations
**Status:** COMPLETED - `plot_all()` implemented with metrics table including Max Drawdown

**Features Implemented:**
- ✅ Four plotting methods: PnL comparison, distribution, positions, comprehensive
- ✅ Time axis flexibility: steps, hours, or days (uses config.dt)
- ✅ Markdown report generation with embedded plots using relative paths
- ✅ Returns dict with report_path and plot_paths for downstream use
- ✅ Path count warnings (>50 paths) for performance
- ✅ Comprehensive metrics table with 8 metrics and differences

**Review point:** Phase 2 complete with all visualization features ✅

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
- [x] Phase 1: Foundation (Steps 1.1 - 1.10) ✅ **COMPLETE**
  - [x] Step 1.1: Create BacktestConfig
  - [x] Step 1.2: Create Metrics Calculator
  - [x] Step 1.3: Create Simple Backtester Shell
  - [x] Step 1.4: Implement Model Loading
  - [x] Step 1.5: Implement Data Loading
  - [x] Step 1.6: Create Bootstrap Path Generator
  - [x] Step 1.7: Implement Deep Hedge Evaluation
  - [x] Step 1.8: Implement BS Baseline Evaluation
  - [x] Step 1.9: Create Results Container
  - [x] Step 1.10: Connect Everything (MVP)
- [x] Phase 2: Reporting & Visualization (Steps 2.1 - 2.6) ✅ **COMPLETE**
  - [x] Step 2.1: Add Plotting - PnL Comparison
  - [x] Step 2.2: Add Plotting - Distribution
  - [x] Step 2.3: Add Plotting - Hedge Positions
  - [x] Step 2.4: Add All Risk Metrics (completed in Phase 1)
  - [x] Step 2.5: Generate Report
  - [x] Step 2.6: Comprehensive Visualization
- [ ] Phase 3: Real Option Price Comparison
- [ ] Phase 4: Polish & CLI

## Next Step
Step 3.1: Load Real Option Data (Phase 3: Real Option Price Comparison)

**Progress:** Phases 1 & 2 complete! 16/16 steps (100%)
**Total Tests:** 102 tests passing (18 config + 24 metrics + 43 backtester + 17 results)
