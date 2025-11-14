import torch
from typing import Dict, Tuple, Optional
from pfhedge.nn import (
    Hedger,
    MultiLayerPerceptron,
    ExpectedShortfall,
    EntropicRiskMeasure,
    QuadraticCVaR,
)
from pfhedge.nn.modules.loss import EntropicLoss

# Import custom features to register them with pfhedge
from crypto.features.custom_features import (
    VolatilityChange,
    MoneynessSquared,
)  # noqa: F401


# Default features for deep hedging
# Note: PFHedge expects "expiry_time" for European options
DEFAULT_FEATURES = [
    "log_moneyness",
    "expiry_time",  # Changed from time_to_maturity to match PFHedge expectations
    "volatility",
    "prev_hedge",
    "volatility_change",  # Custom feature - captures vol dynamics
    "moneyness_squared",  # Custom feature - non-linear gamma effects
]


# Model factory registry
_MODEL_REGISTRY = {}


def _register_model(name: str):

    def decorator(func):
        _MODEL_REGISTRY[name.lower()] = func
        return func

    return decorator


@_register_model("mlp")
def _create_mlp(n_layers: int, n_units: "int | list[int]") -> MultiLayerPerceptron:
    if isinstance(n_units, list):
        units_list = n_units
    else:
        units_list = [n_units] * n_layers
    return MultiLayerPerceptron(n_layers=n_layers, n_units=units_list)


@_register_model("enhanced_mlp")
def _create_enhanced_mlp(
    n_layers: int, n_units: "int | list[int]", dropout: float = 0.15
):
    from .enhanced_mlp import EnhancedMLP

    if isinstance(n_units, list):
        units_list = n_units
    else:
        units_list = [n_units] * n_layers
    return EnhancedMLP(
        n_layers=n_layers, n_units=units_list, dropout=dropout, use_layer_norm=True
    )


@_register_model("lstm")
def _create_lstm(
    n_layers: int, n_units: "int | list[int]", in_features: int = 4
) -> "LongShortTermMemory":
    from .long_short_term_memory import LongShortTermMemory

    hidden_size = n_units if isinstance(n_units, int) else n_units[0]
    return LongShortTermMemory(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=n_layers,
        dropout=0.2 if n_layers > 1 else 0.0,
    )


@_register_model("gru")
def _create_gru(
    n_layers: int, n_units: "int | list[int]", in_features: int = 4
) -> "GatedRecurrentUnit":
    from .gated_recurrent_unit import GatedRecurrentUnit

    hidden_size = n_units if isinstance(n_units, int) else n_units[0]
    return GatedRecurrentUnit(
        in_features=in_features,
        hidden_size=hidden_size,
        num_layers=n_layers,
        dropout=0.2 if n_layers > 1 else 0.0,
    )


# Risk measure factory registry
_CRITERION_REGISTRY = {
    "expected_shortfall": lambda param: ExpectedShortfall(p=param),
    "entropic": lambda param: EntropicRiskMeasure(a=param),
    "entropic_loss": lambda param: EntropicLoss(a=param),
    "quadratic_cvar": lambda param: QuadraticCVaR(lam=param),
}


def create_deep_hedger(
    model_type: str = "mlp",
    n_layers: int = 3,
    n_units: "int | list[int]" = 64,
    risk_measure: str = "expected_shortfall",
    risk_param: float = 0.5,
    features: list = None,
) -> Hedger:
    if features is None:
        features = DEFAULT_FEATURES

    # Create model using registry
    model_type = model_type.lower()
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Available options: {available}"
        )

    model_creator = _MODEL_REGISTRY[model_type]

    # For LSTM/GRU, pass in_features based on number of features
    if model_type in ["lstm", "gru"]:
        n_features = len(features)
        model = model_creator(
            n_layers=n_layers, n_units=n_units, in_features=n_features
        )
    else:
        model = model_creator(n_layers=n_layers, n_units=n_units)

    # Create criterion using registry
    risk_measure = risk_measure.lower()
    if risk_measure not in _CRITERION_REGISTRY:
        available = ", ".join(sorted(_CRITERION_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported risk_measure: '{risk_measure}'. "
            f"Available options: {available}"
        )

    criterion_creator = _CRITERION_REGISTRY[risk_measure]
    criterion = criterion_creator(risk_param)

    # Create hedger
    return Hedger(model=model, inputs=features, criterion=criterion)


def apply_no_trade_band(
    positions: torch.Tensor,
    band_width: float,
) -> torch.Tensor:
    if band_width <= 0:
        return positions

    filtered = torch.zeros_like(positions)
    filtered[:, 0] = positions[:, 0]

    for t in range(1, positions.size(1)):
        prev_pos = filtered[:, t - 1]
        target_pos = positions[:, t]
        change = torch.abs(target_pos - prev_pos)

        filtered[:, t] = torch.where(
            change > band_width,
            target_pos,
            prev_pos,
        )

    return filtered


def calculate_bs_hedge_pnl(
    spots: torch.Tensor,
    bs_delta: torch.Tensor,
    payoffs: torch.Tensor,
    cost: float,
    funding_rate: Optional[torch.Tensor] = None,
    funding_times: Optional[torch.Tensor] = None,
    band_width: float = 0.0,
) -> torch.Tensor:
    bs_delta = apply_no_trade_band(bs_delta, band_width)

    # Capital gains: δ_{i-1} * (S_i - S_{i-1})
    # Use PREVIOUS position (not current) for price changes
    capital_gains = torch.cat(
        [
            torch.zeros_like(spots[:, [0]]),  # No gain at first step
            bs_delta[:, :-1]
            * (spots[:, 1:] - spots[:, :-1]),  # Previous delta * price change
        ],
        dim=1,
    )

    # Cumulative capital gains
    cumulative_pnl = capital_gains.cumsum(dim=1)

    # Subtract payoff at maturity
    cumulative_pnl[:, -1] -= payoffs

    # Transaction costs
    if cost > 0:
        # Position changes: |δ_i - δ_{i-1}|
        delta_changes = torch.cat(
            [
                bs_delta[:, [0]],  # Initial position
                bs_delta[:, 1:] - bs_delta[:, :-1],  # Rebalancing
            ],
            dim=1,
        )

        # Transaction costs applied to spot prices AFTER trade
        # First cost uses initial spot, subsequent costs use new spots
        transaction_costs = cost * torch.abs(delta_changes * spots)

        # Cumulative transaction costs
        cumulative_costs = transaction_costs.cumsum(dim=1)

        # Subtract costs from PnL
        cumulative_pnl -= cumulative_costs

    # Subtract funding costs if provided
    if funding_rate is not None and funding_times is not None:
        cumulative_funding = compute_funding_cum_cost(
            spots=spots,
            positions=bs_delta,
            funding_rate=funding_rate,
            funding_times=funding_times,
        )
        cumulative_pnl = cumulative_pnl - cumulative_funding

    # Calculate hedging efficiency metrics for diagnostics
    if (
        hasattr(calculate_bs_hedge_pnl, "enable_diagnostics")
        and calculate_bs_hedge_pnl.enable_diagnostics
    ):
        # Track position turnover
        position_changes = torch.cat(
            [bs_delta[:, [0]], torch.abs(bs_delta[:, 1:] - bs_delta[:, :-1])], dim=1
        )
        total_turnover = position_changes.sum(dim=1)
        max_position = torch.abs(bs_delta).max(dim=1)[0]

        # Store as attributes for analysis
        cumulative_pnl._turnover = total_turnover
        cumulative_pnl._max_position = max_position

    return cumulative_pnl


def compute_funding_cum_cost(
    spots: torch.Tensor,
    positions: torch.Tensor,
    funding_rate: torch.Tensor,
    funding_times: torch.Tensor,
) -> torch.Tensor:
    if funding_times.dim() == 1:
        funding_times = funding_times.unsqueeze(0).expand_as(spots)

    # Do not charge funding at t=0
    funding_times = funding_times.clone()
    funding_times[..., 0] = False

    payments = positions * funding_rate * spots
    payments = payments * funding_times.to(spots.dtype)

    cumulative_funding = payments.cumsum(dim=1)
    return cumulative_funding


def compare_hedge_performance(
    deep_pnl: torch.Tensor,
    bs_pnl: torch.Tensor,
    names: Tuple[str, str] = ("Deep Hedge", "Black-Scholes"),
) -> Dict[str, Dict[str, float]]:
    # Get final PnL (last time step)
    deep_final = deep_pnl[:, -1] if deep_pnl.dim() == 2 else deep_pnl
    bs_final = bs_pnl[:, -1] if bs_pnl.dim() == 2 else bs_pnl

    # Calculate metrics
    results = {
        names[0]: {
            "mean": deep_final.mean().item(),
            "std": deep_final.std().item(),
            "min": deep_final.min().item(),
            "max": deep_final.max().item(),
        },
        names[1]: {
            "mean": bs_final.mean().item(),
            "std": bs_final.std().item(),
            "min": bs_final.min().item(),
            "max": bs_final.max().item(),
        },
    }

    # Add Sharpe ratios
    for name, pnl in [(names[0], deep_final), (names[1], bs_final)]:
        mean = results[name]["mean"]
        std = results[name]["std"]
        results[name]["sharpe"] = mean / std if std > 0 else 0.0

    return results


def print_performance_comparison(results: Dict[str, Dict[str, float]]) -> None:
    names = list(results.keys())

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<20} {names[0]:>15} {names[1]:>15}")
    print("-" * 52)

    print(
        f"{'Mean PnL':<20} ${results[names[0]]['mean']:>14.2f} ${results[names[1]]['mean']:>14.2f}"
    )
    print(
        f"{'PnL Std':<20} ${results[names[0]]['std']:>14.2f} ${results[names[1]]['std']:>14.2f}"
    )
    print(
        f"{'Sharpe Ratio':<20} {results[names[0]]['sharpe']:>15.3f} {results[names[1]]['sharpe']:>15.3f}"
    )
    print(
        f"{'Min PnL':<20} ${results[names[0]]['min']:>14.2f} ${results[names[1]]['min']:>14.2f}"
    )
    print(
        f"{'Max PnL':<20} ${results[names[0]]['max']:>14.2f} ${results[names[1]]['max']:>14.2f}"
    )

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")

    deep_std = results[names[0]]["std"]
    bs_std = results[names[1]]["std"]

    if deep_std < bs_std:
        print(f"✅ {names[0]} achieves {(1 - deep_std/bs_std)*100:.1f}% lower risk")
    else:
        print(f"⚠️  {names[1]} has {(1 - bs_std/deep_std)*100:.1f}% lower risk")

    deep_sharpe = results[names[0]]["sharpe"]
    bs_sharpe = results[names[1]]["sharpe"]

    if deep_sharpe > bs_sharpe and bs_sharpe != 0:
        print(
            f"✅ {names[0]} has {((deep_sharpe/bs_sharpe - 1)*100):.1f}% better Sharpe ratio"
        )
    elif bs_sharpe > deep_sharpe and deep_sharpe != 0:
        print(
            f"⚠️  {names[1]} has {((bs_sharpe/deep_sharpe - 1)*100):.1f}% better Sharpe ratio"
        )
    elif bs_sharpe > deep_sharpe:
        print(f"⚠️  {names[1]} has better Sharpe ratio")

    deep_mean = results[names[0]]["mean"]
    bs_mean = results[names[1]]["mean"]

    if deep_mean > bs_mean:
        print(f"✅ {names[0]} has ${deep_mean - bs_mean:.2f} higher mean PnL")
    else:
        print(f"⚠️  {names[1]} has ${bs_mean - deep_mean:.2f} higher mean PnL")

    print(f"\n{'='*60}")
