import torch
import pytest
from crypto.strategies.deep_hedge_utils import (
    apply_no_trade_band,
    calculate_bs_hedge_pnl,
)


class TestApplyNoTradeBand:

    def test_zero_band_width_no_filtering(self):
        positions = torch.tensor([[0.5, 0.51, 0.49, 0.52]])
        result = apply_no_trade_band(positions, band_width=0.0)
        assert torch.allclose(result, positions)

    def test_initial_position_always_taken(self):
        positions = torch.tensor([[0.001, 0.002, 0.003]])
        result = apply_no_trade_band(positions, band_width=0.1)
        assert result[0, 0] == 0.001

    def test_small_changes_filtered(self):
        positions = torch.tensor([[0.5, 0.5005, 0.501, 0.51, 0.52]])
        result = apply_no_trade_band(positions, band_width=0.01)

        expected = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.52]])
        assert torch.allclose(result, expected)

    def test_large_changes_executed(self):
        positions = torch.tensor([[0.5, 0.6, 0.7]])
        result = apply_no_trade_band(positions, band_width=0.05)

        assert torch.allclose(result, positions)

    def test_multiple_paths(self):
        positions = torch.tensor(
            [
                [0.5, 0.505, 0.52],
                [0.3, 0.301, 0.32],
            ]
        )
        result = apply_no_trade_band(positions, band_width=0.01)

        expected = torch.tensor(
            [
                [0.5, 0.5, 0.52],
                [0.3, 0.3, 0.32],
            ]
        )
        assert torch.allclose(result, expected)

    def test_cumulative_filtering(self):
        positions = torch.tensor([[0.5, 0.505, 0.510, 0.515, 0.525]])
        result = apply_no_trade_band(positions, band_width=0.01)

        expected = torch.tensor([[0.5, 0.5, 0.5, 0.515, 0.515]])
        assert torch.allclose(result, expected)

    def test_negative_band_width_raises_or_no_filter(self):
        positions = torch.tensor([[0.5, 0.51, 0.49]])
        result = apply_no_trade_band(positions, band_width=-0.01)
        assert torch.allclose(result, positions)

    def test_band_width_edge_case_exact_threshold(self):
        positions = torch.tensor([[0.5, 0.51]])
        result = apply_no_trade_band(positions, band_width=0.01)

        expected = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(result, expected)

    def test_oscillating_positions(self):
        positions = torch.tensor([[0.5, 0.505, 0.495, 0.502, 0.498]])
        result = apply_no_trade_band(positions, band_width=0.01)

        expected = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
        assert torch.allclose(result, expected)


class TestCalculateBsHedgePnlWithBand:

    def test_pnl_with_zero_band(self):
        spots = torch.tensor([[100.0, 101.0, 102.0]])
        bs_delta = torch.tensor([[0.5, 0.5, 0.5]])
        payoffs = torch.tensor([2.0])
        cost = 0.001

        pnl_no_band = calculate_bs_hedge_pnl(
            spots, bs_delta, payoffs, cost, band_width=0.0
        )
        pnl_with_band = calculate_bs_hedge_pnl(
            spots, bs_delta, payoffs, cost, band_width=0.0
        )

        assert torch.allclose(pnl_no_band, pnl_with_band)

    def test_pnl_with_band_reduces_trades(self):
        spots = torch.tensor([[100.0, 100.1, 100.2, 100.3]])
        bs_delta = torch.tensor([[0.5, 0.501, 0.502, 0.503]])
        payoffs = torch.tensor([0.0])
        cost = 0.001

        pnl_no_band = calculate_bs_hedge_pnl(
            spots, bs_delta, payoffs, cost, band_width=0.0
        )
        pnl_with_band = calculate_bs_hedge_pnl(
            spots, bs_delta, payoffs, cost, band_width=0.01
        )

        assert torch.all(pnl_with_band[:, -1] > pnl_no_band[:, -1])

    def test_pnl_shape_unchanged(self):
        n_paths = 5
        n_steps = 10
        spots = torch.randn(n_paths, n_steps).abs() * 100 + 100
        bs_delta = torch.randn(n_paths, n_steps).abs() * 0.5
        payoffs = torch.randn(n_paths).abs()
        cost = 0.001

        pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, cost, band_width=0.01)

        assert pnl.shape == (n_paths, n_steps)


class TestNoTradeBandIntegration:

    def test_band_preserves_pnl_shape(self):
        torch.manual_seed(42)
        n_paths = 10
        n_steps = 20

        spots = torch.randn(n_paths, n_steps).abs() * 1000 + 100000
        bs_delta = torch.randn(n_paths, n_steps).abs() * 0.5 + 0.3
        bs_delta = bs_delta.clamp(0, 1)
        payoffs = torch.maximum(spots[:, -1] - 105000, torch.zeros(n_paths))
        cost = 0.0005

        pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, cost, band_width=0.001)

        assert pnl.shape == (n_paths, n_steps)
        assert torch.all(torch.isfinite(pnl))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
