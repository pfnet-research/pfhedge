from typing import List
from typing import Type

from ._base import Feature
from ._getter import FeatureFactory
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

for cls in FEATURES:
    FeatureFactory().register_feature(str(cls()), cls)
