import pytest

from pfhedge.instruments import Instrument
from pfhedge.instruments import Primary


class MyInstrument(Instrument):
    pass


class MyPrimary(Primary):
    def simulate(self):
        pass


def test_primary_deprecated():
    with pytest.raises(DeprecationWarning):
        _ = MyPrimary()
