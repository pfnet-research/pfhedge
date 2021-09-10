from typing import Callable
from typing import List
from typing import Mapping
from typing import Type
from typing import Union

from ._base import Feature
from .features import Empty
from .features import ExpiryTime
from .features import LogMoneyness
from .features import MaxLogMoneyness
from .features import MaxMoneyness
from .features import Moneyness
from .features import PrevHedge
from .features import Spot
from .features import TimeToMaturity
from .features import UnderlierSpot
from .features import Variance
from .features import Volatility
from .features import Zeros

FEATURES: List[Type[Feature]] = [
    Empty,
    ExpiryTime,
    TimeToMaturity,
    LogMoneyness,
    MaxLogMoneyness,
    MaxMoneyness,
    Moneyness,
    PrevHedge,
    Variance,
    Volatility,
    Zeros,
    Spot,
    UnderlierSpot,
]

DICT_FEATURES: Mapping[str, Type[Feature]] = {str(f()): f for f in FEATURES}


def get_feature_class(feature_class: Union[str, Type[Feature]]) -> Type[Feature]:
    """Get feature class from name.

    Args:
        name (str): Name of feature.

    Returns:
        Feature class
    """
    if isinstance(feature_class, str):
        if feature_class not in DICT_FEATURES:
            raise ValueError(
                f"{feature_class} is not a valid value. "
                "Use sorted(pfhedge.features.FEATURES) to get valid options."
            )
        feature_class = DICT_FEATURES[feature_class]
    elif not issubclass(feature_class, Feature):
        raise TypeError(f"{feature_class} is not Feature.")
    return feature_class


def get_feature(feature: Union[str, Feature]) -> Feature:
    """Get feature from name.

    Args:
        name (str): Name of feature.

    Returns:
        Feature
    """
    if isinstance(feature, str):
        feature = get_feature_class(feature)()
    elif not isinstance(feature, Feature):
        raise TypeError(f"{feature} is not an instance of Feature.")
    return feature
