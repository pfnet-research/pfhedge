import pytest
import torch

from pfhedge.instruments import BrownianStock


class TestBrownianStock:
    """
    pfhedge.instruments.BrownianStock
    """

    def test_repr(self):
        s = BrownianStock(dt=1 / 100)
        assert repr(s) == "BrownianStock(volatility=2.00e-01, dt=1.00e-02)"
        s = BrownianStock(dt=1 / 100, cost=0.001)
        assert (
            repr(s) == "BrownianStock(volatility=2.00e-01, cost=1.00e-03, dt=1.00e-02)"
        )
        s = BrownianStock(dt=1 / 100, dtype=torch.float64)
        assert (
            repr(s)
            == "BrownianStock(volatility=2.00e-01, dt=1.00e-02, dtype=torch.float64)"
        )
        s = BrownianStock(dt=1 / 100, device="cuda:0")
        assert (
            repr(s)
            == "BrownianStock(volatility=2.00e-01, dt=1.00e-02, device='cuda:0')"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        s = BrownianStock(dtype=dtype)
        s.simulate()
        assert s.spot.dtype == dtype

        s = BrownianStock().to(dtype=dtype)
        s.simulate()
        assert s.spot.dtype == dtype

    def test_device(self):
        ...

    def test_to(self):
        s = BrownianStock()
        s.to(device="cuda:0")
        assert s.device == torch.device("cuda:0")
        s.to(device="cuda:1")
        assert s.device == torch.device("cuda:1")
        s.to(dtype=torch.float32)
        assert s.dtype == torch.float32
        s.to(dtype=torch.float64)
        assert s.dtype == torch.float64

        s = BrownianStock()
        s.simulate()
        s.to(dtype=torch.float32)
        assert s.spot.dtype == torch.float32
        s.to(dtype=torch.float64)
        assert s.spot.dtype == torch.float64

    def test_to_error(self):
        with pytest.raises(TypeError):
            BrownianStock().to(dtype=torch.int32)
