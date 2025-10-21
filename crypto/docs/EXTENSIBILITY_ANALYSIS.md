# Training System Extensibility Analysis

**Date**: 2025-10-20
**Status**: Current system is tightly coupled to Brownian motion + European options
**Extensibility Rating**: ⭐⭐⭐⭐☆☆☆ (4/7)

---

## Current Architecture

```
TrainingConfig (hardcoded params)
    ↓
Trainer.create_training_option() (hardcoded call)
    ↓
create_bitcoin_option_from_config() (hardcoded factory)
    ↓
BitcoinPerpetualBrownian + BitcoinEuropeanOption (hardcoded types)
```

### What's Good ✅

1. **PFHedge foundation**: Built on extensible abstractions
   - `Primary` class can be extended for any stochastic process
   - `Derivative` class can be extended for any payoff
   - `Hedger` model is agnostic to instrument types

2. **Clean separation**: Training vs Backtesting are properly separated

3. **Existing infrastructure**:
   - `BitcoinPerpetualBase` provides good base class
   - `BitcoinEuropeanOption` follows PFHedge patterns
   - Model architecture is flexible

### What's Problematic ❌

1. **Hardcoded option creation**:
   ```python
   # crypto/instruments/bitcoin_european_option.py
   def create_bitcoin_option_from_config(config):
       underlier = BitcoinPerpetualBrownian(...)  # HARDCODED!
       option = BitcoinEuropeanOption(...)         # HARDCODED!
   ```

2. **Inflexible TrainingConfig**:
   ```python
   @dataclass
   class TrainingConfig:
       strike: float      # Only for European
       call: bool         # What about barriers/Asians?
       # Missing: barrier_level, averaging_window, jump params, etc.
   ```

3. **No dependency injection**:
   ```python
   # Trainer always calls same hardcoded function
   option, _ = create_bitcoin_option_from_config(config)
   # Can't inject custom option factory!
   ```

---

## Extension Difficulty Matrix

| Extension Type | Difficulty | Time Estimate | Notes |
|---------------|-----------|---------------|-------|
| **New Stochastic Process** | ⭐⭐ Easy | 1-2 days | Good abstraction via `BitcoinPerpetualBase` |
| - Add Heston model | ⭐ Very Easy | 4 hours | PFHedge already has Heston |
| - Add Jump Diffusion (Merton) | ⭐⭐ Easy | 1 day | Just implement `simulate()` |
| - Add Local Volatility | ⭐⭐⭐ Medium | 2-3 days | Need calibration logic |
| **New Vanilla Options** | ⭐⭐⭐ Medium | 2-3 days | Need config refactoring |
| - Add Asian options | ⭐⭐⭐ Medium | 2 days | New payoff + config params |
| - Add Lookback options | ⭐⭐⭐ Medium | 2 days | Track max/min in payoff |
| **Exotic Options** | ⭐⭐⭐⭐⭐⭐ Hard | 1-2 weeks | Major refactoring needed |
| - Add Barrier options | ⭐⭐⭐⭐⭐ Hard | 1 week | Monitoring + config explosion |
| - Add Snowball options | ⭐⭐⭐⭐⭐⭐ Very Hard | 2 weeks | Complex payoff + early termination |

---

## Recommended Refactoring (Priority: Future Work)

### Phase 1: Factory Pattern (1 day)

**Goal**: Decouple option creation from hardcoded types

```python
# crypto/instruments/factory.py (NEW FILE)

from typing import Protocol, Dict, Any

class OptionFactory(Protocol):
    """Protocol for option factories."""
    def create(self, config: Dict[str, Any]) -> BitcoinEuropeanOption:
        ...

class EuropeanOptionFactory:
    """Factory for European options."""

    def create(self, config: Dict[str, Any]):
        # Choose underlier type
        underlier_type = config.get("underlier_type", "brownian")

        if underlier_type == "brownian":
            underlier = BitcoinPerpetualBrownian(...)
        elif underlier_type == "heston":
            underlier = BitcoinPerpetualHeston(...)
        elif underlier_type == "merton":
            underlier = BitcoinPerpetualMerton(...)
        else:
            raise ValueError(f"Unknown underlier type: {underlier_type}")

        # Create option
        return BitcoinEuropeanOption(underlier=underlier, ...)

class AsianOptionFactory:
    """Factory for Asian options."""
    def create(self, config: Dict[str, Any]):
        # Similar pattern
        return BitcoinAsianOption(...)

# Registry for easy extension
OPTION_FACTORIES = {
    "european": EuropeanOptionFactory(),
    "asian": AsianOptionFactory(),
    # Easy to add more!
}

def create_option_from_config(config: Dict[str, Any]):
    """Factory dispatcher."""
    option_type = config.get("option_type", "european")
    factory = OPTION_FACTORIES.get(option_type)

    if factory is None:
        raise ValueError(f"Unknown option type: {option_type}")

    return factory.create(config)
```

**Benefits**:
- ✅ Easy to add new option types (just register in dict)
- ✅ Easy to add new underlier types (switch in factory)
- ✅ No changes to Trainer code
- ✅ Backward compatible

---

### Phase 2: Config Composition (2 days)

**Goal**: Split monolithic `TrainingConfig` into composable sub-configs

```python
# crypto/training/config.py (REFACTORED)

from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class UnderlierConfig:
    """Base config for underliers."""
    type: str  # "brownian", "heston", "merton", etc.
    cost: float = 0.0005
    dt_hours: float = 8.0

@dataclass
class BrownianConfig(UnderlierConfig):
    """Config for Brownian motion."""
    type: str = "brownian"
    volatility: float = 0.8
    drift: float = 0.0

@dataclass
class HestonConfig(UnderlierConfig):
    """Config for Heston model."""
    type: str = "heston"
    kappa: float = 1.0      # Mean reversion speed
    theta: float = 0.04     # Long-term variance
    sigma: float = 0.5      # Vol of vol
    rho: float = -0.7       # Correlation

@dataclass
class OptionConfig:
    """Base config for options."""
    type: str  # "european", "asian", "barrier", etc.
    strike: float
    maturity_days: int
    call: bool = True

@dataclass
class EuropeanConfig(OptionConfig):
    """Config for European options."""
    type: str = "european"
    # No extra params needed

@dataclass
class AsianConfig(OptionConfig):
    """Config for Asian options."""
    type: str = "asian"
    averaging_start: int = 0  # Start averaging from step N

@dataclass
class BarrierConfig(OptionConfig):
    """Config for Barrier options."""
    type: str = "barrier"
    barrier_level: float
    barrier_type: str = "up_and_out"  # "up_and_out", "down_and_out", etc.

@dataclass
class TrainingConfig:
    """Refactored training config using composition."""

    # Core training parameters (always needed)
    n_paths: int = 10000
    n_epochs: int = 80
    train_seed: int = 42
    test_seed: int = 43
    test_n_paths: int = 200

    # Model architecture
    n_layers: int = 4
    n_units: int = 128
    risk_measure: str = "expected_shortfall"
    risk_param: float = 0.9

    # Composed sub-configs (extensible!)
    underlier_config: Union[BrownianConfig, HestonConfig]
    option_config: Union[EuropeanConfig, AsianConfig, BarrierConfig]

    # Paths
    model_path: str = "models/deep_hedger.pth"
    output_dir: str = "training_results"
    device: str = "cpu"

# Example usage:
brownian = BrownianConfig(volatility=0.8, drift=0.0, cost=0.0005)
european = EuropeanConfig(strike=50000, maturity_days=14, call=True)

config = TrainingConfig(
    n_epochs=100,
    underlier_config=brownian,
    option_config=european,
)
```

**Benefits**:
- ✅ Clear separation of concerns
- ✅ Type-safe configs (mypy can check)
- ✅ Easy to add new option/underlier types
- ✅ No parameter explosion in main config
- ✅ Supports complex combinations

**Migration Path**:
```python
# Keep old API for backward compatibility
@classmethod
def from_legacy(cls,
                strike: float,
                maturity_days: int,
                volatility: float = 0.8,
                **kwargs):
    """Create from old-style parameters."""
    underlier = BrownianConfig(volatility=volatility, ...)
    option = EuropeanConfig(strike=strike, maturity_days=maturity_days)
    return cls(underlier_config=underlier, option_config=option, **kwargs)
```

---

### Phase 3: Dependency Injection (1 day)

**Goal**: Allow custom option factories to be injected into Trainer

```python
# crypto/training/trainer.py (REFACTORED)

from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from crypto.instruments import BitcoinEuropeanOption

class Trainer:
    """Train deep hedging models with configurable option factory."""

    def __init__(
        self,
        config: TrainingConfig,
        option_factory: Optional[Callable] = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            option_factory: Optional custom factory function.
                           Signature: (config) -> (train_option, test_option)
                           If None, uses default factory from config.
        """
        self.config = config
        self.option_factory = option_factory or self._default_option_factory

        # Placeholders
        self.train_option = None
        self.test_option = None
        self.model = None

    def _default_option_factory(self, n_paths: int, seed: int):
        """Default factory using config."""
        from crypto.instruments.factory import create_option_from_config

        # Build config dict from composed configs
        config_dict = {
            "option_type": self.config.option_config.type,
            "underlier_type": self.config.underlier_config.type,
            "n_paths": n_paths,
            "seed": seed,
            **self.config.option_config.__dict__,
            **self.config.underlier_config.__dict__,
        }

        return create_option_from_config(config_dict)

    def create_training_option(self) -> "BitcoinEuropeanOption":
        """Create training option using configured factory."""
        print("Creating training option...")

        # Use injected or default factory
        option = self.option_factory(
            n_paths=self.config.n_paths,
            seed=self.config.train_seed,
        )

        self.train_option = option
        return option

    def create_test_option(self) -> "BitcoinEuropeanOption":
        """Create test option using configured factory."""
        print("Creating test option...")

        option = self.option_factory(
            n_paths=self.config.test_n_paths,
            seed=self.config.test_seed,
        )

        self.test_option = option
        return option
```

**Usage Examples**:

```python
# Example 1: Default usage (backward compatible)
config = TrainingConfig(...)
trainer = Trainer(config)  # Uses default factory

# Example 2: Custom factory for exotic option
def my_snowball_factory(n_paths, seed):
    """Custom factory for snowball options."""
    underlier = BitcoinPerpetualHeston(...)  # Custom underlier
    option = BitcoinSnowballOption(...)       # Custom option
    return option

config = TrainingConfig(...)
trainer = Trainer(config, option_factory=my_snowball_factory)
# Now trainer uses your custom factory!

# Example 3: Lambda for quick customization
trainer = Trainer(
    config,
    option_factory=lambda n, s: create_custom_option(n, s, extra_param=True)
)
```

**Benefits**:
- ✅ Fully extensible without modifying Trainer
- ✅ Backward compatible (default factory)
- ✅ Testable (easy to inject mock factory)
- ✅ Supports one-off experiments

---

## Implementation Roadmap

### Quick Wins (Can do today)
- [ ] Fix `compare_hedge_performance` key mismatch bug
- [ ] Add type hints to option creation functions
- [ ] Document current coupling in code comments

### Short Term (1-2 weeks)
- [ ] Phase 1: Implement Factory Pattern
- [ ] Phase 2: Refactor TrainingConfig with composition
- [ ] Phase 3: Add dependency injection to Trainer
- [ ] Add unit tests for new factories
- [ ] Update documentation and examples

### Medium Term (1-2 months)
- [ ] Add Heston stochastic volatility underlier
- [ ] Add Asian option support
- [ ] Add Jump Diffusion (Merton) underlier
- [ ] Create example notebooks for each

### Long Term (3-6 months)
- [ ] Add Barrier options support
- [ ] Add Local Volatility calibration
- [ ] Add Snowball options (complex exotic)
- [ ] Multi-asset options support

---

## Example: Adding Heston Model (After Refactor)

**Step 1**: Create Heston underlier class (4 hours)
```python
# crypto/instruments/bitcoin_perpetual_heston.py
class BitcoinPerpetualHeston(BitcoinPerpetualBase):
    def __init__(self, kappa, theta, sigma, rho, ...):
        # Implement Heston stochastic volatility
        pass
```

**Step 2**: Add Heston config (30 minutes)
```python
# Already done in Phase 2!
heston_config = HestonConfig(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7)
```

**Step 3**: Register in factory (5 minutes)
```python
# crypto/instruments/factory.py
if underlier_type == "heston":
    underlier = BitcoinPerpetualHeston(
        kappa=config['kappa'],
        theta=config['theta'],
        ...
    )
```

**Step 4**: Use it! (1 minute)
```python
config = TrainingConfig(
    underlier_config=HestonConfig(kappa=1.0, theta=0.04),
    option_config=EuropeanConfig(strike=50000),
    n_epochs=100,
)
trainer = Trainer(config)
results = trainer.train()
```

**Total time**: ~5 hours (vs 2-3 days currently!)

---

## Conclusion

**Current State**:
- System works well for Brownian + European
- Foundation is solid (PFHedge abstractions)
- Top-level pipeline is tightly coupled

**After Refactor**:
- Adding new stochastic processes: **Hours** instead of days
- Adding new option types: **Days** instead of weeks
- System becomes production-ready for multi-instrument trading

**Effort**: ~4 days of refactoring work for massive long-term benefits

**Priority**: Medium (current system works, but this unblocks future research)

---

## References

- PFHedge documentation: https://pfhedge.readthedocs.io/
- Factory Pattern: https://refactoring.guru/design-patterns/factory-method
- Dependency Injection: https://en.wikipedia.org/wiki/Dependency_injection
- Heston Model: Heston (1993) "A Closed-Form Solution for Options with Stochastic Volatility"
- Merton Jump Diffusion: Merton (1976) "Option pricing when underlying stock returns are discontinuous"
