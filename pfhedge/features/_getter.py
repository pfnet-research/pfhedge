from typing import Dict
from typing import Union

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
    TimeToMaturity(),
    LogMoneyness(),
    MaxLogMoneyness(),
    MaxMoneyness(),
    Moneyness(),
    PrevHedge(),
    Volatility(),
    Zeros(),
]


def get_feature(feature: Union[str, Feature]) -> Feature:
    """Get feature from name.

    Args:
        name (str): Name of feature.

    Returns:
        Feature
    """
    dict_features: Dict[str, Feature] = {str(f): f for f in FEATURES}

    if isinstance(feature, str):
        if feature not in dict_features:
            raise ValueError(
                f"{feature} is not a valid value. "
                "Use sorted(pfhedge.features.FEATURES) to get valid options."
            )
        feature = dict_features[feature]
    elif not isinstance(feature, Feature):
        raise TypeError(f"{feature} is not an instance of Feature.")
    return feature
