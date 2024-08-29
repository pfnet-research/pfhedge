from math import sqrt

import pytest
import torch
from torch.testing import assert_close

from pfhedge.stochastic import generate_merton_jump
from pfhedge.stochastic.engine import RandnSobolBoxMuller


class TestGenerateMertonJumpStock:
    func = staticmethod(generate_merton_jump)

    def setup_method(self):
        self.jump_test_func = self.func

    def test_generate_brownian_mean_no_jump(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250

        output = self.jump_test_func(
            n_paths, n_steps, jump_std=0.0, device=torch.device(device)
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result)
        std = 0.2 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

    @pytest.mark.gpu
    def test_generate_brownian_mean_no_jump_gpu(self):
        self.test_generate_brownian_mean_no_jump(device="cuda")

    def test_generate_brownian_mean_no_jump1(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250

        output = self.jump_test_func(n_paths, n_steps, jump_per_year=0.0, device=device)
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result)
        std = 0.2 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

    @pytest.mark.gpu
    def test_generate_brownian_mean_no_jump1_gpu(self):
        self.test_generate_brownian_mean_no_jump1(device="cuda")

    def test_generate_brownian_mean_no_jump_std(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250

        output = self.jump_test_func(
            n_paths,
            n_steps,
            jump_per_year=68.2,  # default value
            jump_std=0.0,
            jump_mean=0.1,
            device=torch.device(device),
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result)
        std = 0.5 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

    @pytest.mark.gpu
    def test_generate_brownian_mean_no_jump_std_gpu(self):
        self.test_generate_brownian_mean_no_jump_std(device="cuda")

    def test_generate_brownian_mean(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250

        output = self.jump_test_func(
            n_paths, n_steps, jump_per_year=1, device=torch.device(device)
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result)
        std = 0.2 * sqrt(1 / n_paths) + 0.3 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

    @pytest.mark.gpu
    def test_generate_brownian_mean_gpu(self):
        self.test_generate_brownian_mean(device="cuda")

    def test_generate_jump_nosigma(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_steps = 250

        result = self.jump_test_func(
            1, n_steps, sigma=0, jump_per_year=0, device=torch.device(device)
        )
        expect = torch.ones(1, n_steps).to(device)
        assert_close(result, expect)

        mu = 0.1
        dt = 0.01
        result = self.jump_test_func(
            1,
            n_steps,
            mu=mu,
            sigma=0,
            dt=dt,
            jump_per_year=0,
            device=torch.device(device),
        ).log()
        expect = (
            torch.linspace(0, mu * dt * (n_steps - 1), n_steps).unsqueeze(0).to(device)
        )
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_generate_jump_nosigma_gpu(self):
        self.test_generate_jump_nosigma(device="cpu")

    def test_generate_jump_nosigma2(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_steps = 250

        result = self.jump_test_func(
            1, n_steps, sigma=0, jump_std=0, device=torch.device(device)
        )
        expect = torch.ones(1, n_steps).to(device)
        assert_close(result, expect)

        mu = 0.1
        dt = 0.01
        result = self.jump_test_func(
            1, n_steps, mu=mu, sigma=0, dt=dt, jump_std=0, device=torch.device(device)
        ).log()
        expect = (
            torch.linspace(0, mu * dt * (n_steps - 1), n_steps).unsqueeze(0).to(device)
        )
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_generate_jump_nosigma2_gpu(self):
        self.test_generate_jump_nosigma2(device="cuda")

    def test_generate_jump_std(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250

        output = self.jump_test_func(
            n_paths, n_steps, jump_per_year=0, device=torch.device(device)
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].log().std()
        expect = torch.full_like(result, 0.2)
        assert_close(result, expect, atol=0, rtol=0.1)

    @pytest.mark.gpu
    def test_generate_jump_std_gpu(self):
        self.test_generate_jump_std(device="cuda")

    def test_generate_jump_std2(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250

        output = self.jump_test_func(
            n_paths, n_steps, jump_std=0, device=torch.device(device)
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].log().std()
        expect = torch.full_like(result, 0.2)
        assert_close(result, expect, atol=0, rtol=0.1)

    @pytest.mark.gpu
    def test_generate_jump_std2_gpu(self):
        self.test_generate_jump_std2(device="cuda")

    def test_generate_jump_mean_init_state(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250

        output = self.jump_test_func(
            n_paths,
            n_steps,
            init_state=1.0,
            jump_per_year=0,
            device=torch.device(device),
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result)
        std = 0.2 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

        output = self.jump_test_func(
            n_paths,
            n_steps,
            init_state=torch.tensor(1.0),
            jump_per_year=0,
            device=torch.device(device),
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result)
        std = 0.2 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

        output = self.jump_test_func(
            n_paths,
            n_steps,
            init_state=torch.tensor([1.0]),
            jump_per_year=0,
            device=torch.device(device),
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result)
        std = 0.2 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

    @pytest.mark.gpu
    def test_generate_jump_mean_init_state_gpu(self):
        self.test_generate_jump_mean_init_state(device="cuda")

    def test_generate_jump_mean_mu(self, device: str = "cpu"):
        torch.manual_seed(42)
        n_paths = 10000
        n_steps = 250
        dt = 1 / 250
        mu = 0.1

        output = self.jump_test_func(
            n_paths, n_steps, mu=mu, jump_per_year=0, device=torch.device(device)
        )
        result = output[:, -1].mean().log()
        expect = torch.full_like(result, mu * dt * n_steps).to(device)
        std = 0.2 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=3 * std, rtol=0)

    @pytest.mark.gpu
    def test_generate_jump_mean_mu_gpu(self):
        self.test_generate_jump_mean_mu(device="cuda")

    def test_generate_jump_dtype(self, device: str = "cpu"):
        torch.manual_seed(42)

        output = self.jump_test_func(
            1, 1, dtype=torch.float32, device=torch.device(device)
        )
        assert output.dtype == torch.float32

        output = self.jump_test_func(
            1, 1, dtype=torch.float64, device=torch.device(device)
        )
        assert output.dtype == torch.float64

    @pytest.mark.gpu
    def test_generate_jump_dtype_gpu(self):
        self.test_generate_jump_dtype(device="cuda")

    def test_generate_jump_sobol_mean(self, device: str = "cpu"):
        n_paths = 10000
        n_steps = 250

        engine = RandnSobolBoxMuller(seed=42, scramble=True)
        output = self.jump_test_func(
            n_paths,
            n_steps,
            engine=engine,
            jump_per_year=0,
            device=torch.device(device),
        )
        assert output.size() == torch.Size((n_paths, n_steps))
        result = output[:, -1].mean()
        expect = torch.ones_like(result).to(device)
        std = 0.2 * sqrt(1 / n_paths)
        assert_close(result, expect, atol=10 * std, rtol=0)

    @pytest.mark.gpu
    def test_generate_jump_sobol_mean_gpu(self):
        self.test_generate_jump_sobol_mean(device="cuda")
