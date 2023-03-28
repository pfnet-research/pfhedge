import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BasePrimary
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption


class NullPrimary(BasePrimary):
    def simulate(self):
        pass


def test_extra_repr_is_empty_by_default():
    assert NullPrimary().extra_repr() == ""


class TestBrownianStock:
    def test_repr(self):
        s = BrownianStock(dt=1 / 100)
        expect = "BrownianStock(sigma=0.2000, dt=0.0100)"
        assert repr(s) == expect

        s = BrownianStock(dt=1 / 100, cost=0.001)
        expect = "BrownianStock(sigma=0.2000, cost=0.0010, dt=0.0100)"
        assert repr(s) == expect

        s = BrownianStock(dt=1 / 100, dtype=torch.float64)
        expect = "BrownianStock(sigma=0.2000, dt=0.0100, dtype=torch.float64)"
        assert repr(s) == expect

        s = BrownianStock(dt=1 / 100, device=torch.device("cuda:0"))
        expect = "BrownianStock(sigma=0.2000, dt=0.0100, device='cuda:0')"
        assert repr(s) == expect

    def test_register_buffer(self):
        s = BrownianStock()
        with pytest.raises(TypeError):
            s.register_buffer(None, torch.zeros(10))
        with pytest.raises(KeyError):
            s.register_buffer("a.b", torch.zeros(10))
        with pytest.raises(KeyError):
            s.register_buffer("", torch.zeros(10))
        with pytest.raises(KeyError):
            s.register_buffer("simulate", torch.zeros(10))
        with pytest.raises(TypeError):
            s.register_buffer("a", torch.nn.ReLU())

    def test_buffers(self, device: str = "cpu"):
        torch.manual_seed(42)
        a = torch.randn(10).to(device)
        b = torch.randn(10).to(device)
        c = torch.randn(10).to(device)

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

    @pytest.mark.gpu
    def test_buffers_gpu(self):
        self.test_buffers(device="cuda")

    def test_buffer_attribute_error(self):
        class MyPrimary(BasePrimary):
            # Primary without super().__init__()
            def __init__(self):
                pass

            def simulate(self):
                self.register_buffer("a", torch.zeros(10))

        with pytest.raises(AttributeError):
            MyPrimary().simulate()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_init_dtype(self, dtype, device: str = "cpu"):
        s = BrownianStock(dtype=dtype, device=device)
        s.simulate()
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_init_dtype_gpu(self, dtype):
        self.test_init_dtype(dtype, device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_to_dtype(self, dtype, device: str = "cpu"):
        # to(dtype) before simulate()
        s = BrownianStock().to(dtype=dtype, device=device)
        s.simulate()
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

        # to(dtype) after simulate()
        s = BrownianStock().to(device)
        s.simulate()
        s.to(dtype=dtype)
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

        # to(instrument) before simulate()
        instrument = BrownianStock(dtype=dtype, device=device)
        s = BrownianStock().to(instrument)
        s.simulate()
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

        instrument = BrownianStock(dtype=dtype, device=device)
        s = BrownianStock().to(instrument=instrument)  # Use kwargs
        s.simulate()
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

        instrument = EuropeanOption(BrownianStock(dtype=dtype, device=device))
        s = BrownianStock().to(instrument)
        s.simulate()
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

        # to(instrument) after simulate()
        instrument = BrownianStock(dtype=dtype, device=device)
        s = BrownianStock()
        s.simulate()
        s.to(instrument)
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

        instrument = EuropeanOption(BrownianStock(dtype=dtype, device=device))
        s = BrownianStock()
        s.simulate()
        s.to(instrument)
        assert s.dtype == dtype
        assert s.spot.dtype == dtype

        # Only accepts float
        with pytest.raises(TypeError):
            BrownianStock().to(dtype=torch.int32)

        # Aliases

        s = BrownianStock().to(device)
        s.simulate()
        s.double()
        assert s.dtype == torch.float64
        assert s.spot.dtype == torch.float64

        s = BrownianStock().to(device)
        s.simulate()
        s.float64()
        assert s.dtype == torch.float64
        assert s.spot.dtype == torch.float64

        s = BrownianStock().to(device)
        s.simulate()
        s.float()
        assert s.dtype == torch.float32
        assert s.spot.dtype == torch.float32

        s = BrownianStock().to(device)
        s.simulate()
        s.float32()
        assert s.dtype == torch.float32
        assert s.spot.dtype == torch.float32

        s = BrownianStock().to(device)
        s.simulate()
        s.half()
        assert s.dtype == torch.float16
        assert s.spot.dtype == torch.float16

        s = BrownianStock().to(device)
        s.simulate()
        s.float16()
        assert s.dtype == torch.float16
        assert s.spot.dtype == torch.float16

        s = BrownianStock().to(device)
        s.simulate()
        s.bfloat16()
        assert s.dtype == torch.bfloat16
        assert s.spot.dtype == torch.bfloat16

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_to_dtype_gpu(self, dtype):
        self.test_to_dtype(dtype, device="cuda")

    def test_simulate_shape(self, device: str = "cpu"):
        s = BrownianStock(dt=0.1).to(device)
        s.simulate(time_horizon=0.2, n_paths=10)
        assert s.spot.size() == torch.Size((10, 3))

        s = BrownianStock(dt=0.1).to(device)
        s.simulate(time_horizon=0.25, n_paths=10)
        assert s.spot.size() == torch.Size((10, 4))

    @pytest.mark.gpu
    def test_simulate_shape_gpu(self):
        self.test_simulate_shape(device="cuda")

    @pytest.mark.parametrize("sigma", [0.2, 0.1])
    def test_volatility(self, sigma, device: str = "cpu"):
        s = BrownianStock(sigma=sigma).to(device)
        s.simulate()
        result = s.volatility
        expect = torch.full_like(s.spot, sigma)
        assert_close(result, expect)

        result = s.variance
        expect = torch.full_like(s.spot, sigma ** 2)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("sigma", [0.2, 0.1])
    def test_volatility_gpu(self, sigma):
        self.test_volatility(sigma, device="cuda")

    def test_init_device(self):
        s = BrownianStock(device=torch.device("cuda:0"))
        assert s.cpu().device == torch.device("cpu")

    def test_to_device(self):
        # to(device)
        s = BrownianStock().to(device="cuda:1")
        assert s.device == torch.device("cuda:1")

        # to(instrument)
        instrument = BrownianStock(device=torch.device("cuda:1"))
        s = BrownianStock().to(instrument)
        assert s.device == torch.device("cuda:1")

        instrument = EuropeanOption(BrownianStock(device="cuda:1"))
        s = BrownianStock().to(instrument)
        assert s.device == torch.device("cuda:1")

        # Aliases

        s = BrownianStock(device=torch.device("cuda:0"))
        assert s.cpu().device == torch.device("cpu")

        s = BrownianStock()
        assert s.cuda().device == torch.device("cuda")

        s = BrownianStock()
        assert s.cuda(1).device == torch.device("cuda:1")

    def test_is_listed(self):
        s = BrownianStock()
        assert s.is_listed
