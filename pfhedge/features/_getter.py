from ._base import Feature
from .features import ExpiryTime
from .features import LogMoneyness
from .features import MaxLogMoneyness
from .features import MaxMoneyness
from .features import Moneyness
from .features import PrevHedge
from .features import Volatility
from .features import Zero

FEATURES = [
    ExpiryTime(),
    LogMoneyness(),
    MaxLogMoneyness(),
    MaxMoneyness(),
    Moneyness(),
    PrevHedge(),
    Volatility(),
    Zero(),
]


def get_feature(feature):
    """Get feature from name.

    Args:
        name (str): Name of feature.

    Returns:
        Feature
    """
    dict_features = {str(f): f for f in FEATURES}

    if isinstance(feature, str):
        f = dict_features.get(feature)
        if f is None:
            raise ValueError(
                f"{feature} is not a valid value. "
                "Use sorted(pfhedge.features.FEATURES) to get valid options."
            )
    else:
        f = feature
        if not isinstance(feature, Feature):
            raise TypeError(f"{feature} is not an instance of Feature.")
    return f
