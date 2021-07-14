from ._base import Feature
from .features import Empty
from .features import ExpiryTime
from .features import LogMoneyness
from .features import MaxLogMoneyness
from .features import MaxMoneyness
from .features import Moneyness
from .features import PrevHedge
from .features import TimeToMaturity
from .features import Volatility
from .features import Zeros

FEATURES = [
    Empty(),
    ExpiryTime(),
    LogMoneyness(),
    MaxLogMoneyness(),
    MaxMoneyness(),
    Moneyness(),
    PrevHedge(),
    TimeToMaturity(),
    Volatility(),
    Zeros(),
]


def get_feature(feature):
    """Get feature from name.

    Args:
        name (str): Name of feature.

    Returns:
        Feature
    """
    # TODO(simaki): Remove this when expiry_time is removed.
    if feature == "expiry_time":
        raise DeprecationWarning(
            "'expiry_time' is deprecated. Use 'time_to_maturity' instead."
        )

    dict_features = {str(f): f for f in FEATURES}

    if isinstance(feature, str):
        if feature not in dict_features:
            raise ValueError(
                f"{feature} is not a valid value. "
                "Use sorted(pfhedge.features.FEATURES) to get valid options."
            )
        feature = dict_features[feature]
    else:
        # If `feature` is Feature object, pass it through.
        if not isinstance(feature, Feature):
            raise TypeError(f"{feature} is not an instance of Feature.")
    return feature
