import pytest
import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import Primary


def test_buffer_attribute_error():
    class MyPrimary(Primary):
        # Primary without super().__init__()
        def __init__(self):
            pass

        def simulate(self):
            self.register_buffer("a", torch.empty(10))

    with pytest.raises(AttributeError):
        MyPrimary().simulate()


class TestBrownianStock:
    """
    pfhedge.instruments.BrownianStock
    """

    def test_repr(self):
        s = BrownianStock(dt=1 / 100)
        assert repr(s) == "BrownianStock(volatility=2.00e-01, dt=1.00e-02)"

        s = BrownianStock(dt=1 / 100, cost=0.001)
        expect = "BrownianStock(volatility=2.00e-01, cost=1.00e-03, dt=1.00e-02)"
        assert repr(s) == expect

        s = BrownianStock(dt=1 / 100, dtype=torch.float64)
        expect = "BrownianStock(volatility=2.00e-01, dt=1.00e-02, dtype=torch.float64)"
        assert repr(s) == expect

        s = BrownianStock(dt=1 / 100, device="cuda:0")
        expect = "BrownianStock(volatility=2.00e-01, dt=1.00e-02, device='cuda:0')"
        assert repr(s) == expect

    def test_register_buffer(self):
        s = BrownianStock()
        with pytest.raises(TypeError):
            s.register_buffer(None, torch.empty(10))
        with pytest.raises(KeyError):
            s.register_buffer("a.b", torch.empty(10))
        with pytest.raises(KeyError):
            s.register_buffer("", torch.empty(10))
        with pytest.raises(KeyError):
            s.register_buffer("simulate", torch.empty(10))
        with pytest.raises(TypeError):
            s.register_buffer("a", torch.nn.ReLU())

    def test_buffers(self):
        torch.manual_seed(42)
        a = torch.randn(10)
        b = torch.randn(10)
        c = torch.randn(10)

        s = BrownianStock()
        s.register_buffer("a", a)
        s.register_buffer("b", b)
        s.register_buffer("c", c)
        result = list(s.named_buffers())
        expect = [("a", a), ("b", b), ("c", c)]
        assert result == expect

        result = list(s.buffers())
        expect = [a, b, c]
        assert result == expect

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        # __init__
        s = BrownianStock(dtype=dtype)
        s.simulate()
        assert s.spot.dtype == dtype

        # to() before simulate
        s = BrownianStock().to(dtype)
        s.simulate()
        assert s.spot.dtype == dtype

        # to() after simulate
        s = BrownianStock()
        s.simulate()
        s.to(dtype)
        assert s.spot.dtype == dtype

        s = BrownianStock()
        s.simulate()
        s.float()
        assert s.dtype == torch.float32
        assert s.spot.dtype == torch.float32

    def test_device(self):
        ...

    def test_to_device(self):
        s = BrownianStock()
        s.to(device="cuda:0")
        assert s.device == torch.device("cuda:0")
        s.to(device="cuda:1")
        assert s.device == torch.device("cuda:1")
        s.to(dtype=torch.float32)
        assert s.dtype == torch.float32
        s.to(dtype=torch.float64)
        assert s.dtype == torch.float64

    def test_to_error(self):
        with pytest.raises(TypeError):
            BrownianStock().to(dtype=torch.int32)
