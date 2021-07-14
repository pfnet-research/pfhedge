import pytest

from pfhedge.features._getter import FEATURES
from pfhedge.features._getter import get_feature


@pytest.mark.parametrize("feature", FEATURES)
def test_get_feature(feature):
    if str(feature) == "expiry_time":
        with pytest.raises(DeprecationWarning):
            get_feature(str(feature))
        return
    assert get_feature(str(feature)).__class__ == feature.__class__
    assert get_feature(feature).__class__ == feature.__class__


def test_get_feature_error():
    with pytest.raises(ValueError):
        get_feature("nonexitent_feature")
    with pytest.raises(TypeError):
        get_feature(0)
