import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanBinaryOption

cls = EuropeanBinaryOption


class TestEuropeanBinaryOption:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), strike=2.0).to(device)
        derivative.underlier.register_buffer(
            "spot",
            torch.tensor(
                [[1.0, 1.0, 1.0, 1.0], [3.0, 1.0, 1.0, 1.0], [1.9, 2.0, 2.1, 3.0]]
            )
            .to(device)
            .T,
        )
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 1.0, 1.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    @pytest.mark.parametrize("volatility", [0.20, 0.10])
    @pytest.mark.parametrize("strike", [1.0, 0.5, 2.0])
    @pytest.mark.parametrize("maturity", [0.1, 1.0])
    @pytest.mark.parametrize("n_paths", [100])
    @pytest.mark.parametrize("init_spot", [1.0, 1.1, 0.9])
    def test_parity(
        self,
        volatility,
        strike,
        maturity,
        n_paths,
        init_spot,
        device: str = "cpu",
    ):
        """
        Test put-call parity.
        """
        stock = BrownianStock(volatility).to(device)
        co = EuropeanBinaryOption(
            stock, strike=strike, maturity=maturity, call=True
        ).to(device)
        po = EuropeanBinaryOption(
            stock, strike=strike, maturity=maturity, call=False
        ).to(device)
        co.simulate(n_paths=n_paths, init_state=(init_spot,))
        po.simulate(n_paths=n_paths, init_state=(init_spot,))

        c = co.payoff()
        p = po.payoff()

        assert (c + p == 1.0).all()

    @pytest.mark.gpu
    @pytest.mark.parametrize("volatility", [0.20, 0.10])
    @pytest.mark.parametrize("strike", [1.0, 0.5, 2.0])
    @pytest.mark.parametrize("maturity", [0.1, 1.0])
    @pytest.mark.parametrize("n_paths", [100])
    @pytest.mark.parametrize("init_spot", [1.0, 1.1, 0.9])
    def test_parity_gpu(self, volatility, strike, maturity, n_paths, init_spot):
        self.test_parity(
            volatility, strike, maturity, n_paths, init_spot, device="cuda"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(dtype=dtype)).to(device)
        assert derivative.dtype == dtype
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = EuropeanBinaryOption(BrownianStock()).to(dtype=dtype).to(device)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    @pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
    def test_device(self, device):
        derivative = EuropeanBinaryOption(BrownianStock(device=device))
        assert derivative.device == torch.device(device)

    def test_repr(self):
        derivative = EuropeanBinaryOption(BrownianStock(), maturity=1.0)
        expect = """\
EuropeanBinaryOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = EuropeanBinaryOption(BrownianStock(), call=False, maturity=1.0)
        expect = """\
EuropeanBinaryOption(
  call=False, strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

    def test_init_dtype_deprecated(self):
        with pytest.raises(DeprecationWarning):
            _ = cls(BrownianStock(), dtype=torch.float64)
