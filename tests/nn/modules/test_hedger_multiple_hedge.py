"""
Test for multiple hedge instruments functionality in Hedger.

This test ensures that the Hedger class properly handles multiple hedge instruments
without encountering size mismatch errors in the P&L calculation.

The issue was that during optimizer configuration, compute_pl was called with default
hedge instruments, but during training it was called with explicitly provided multiple
hedge instruments, causing a tensor size mismatch.
"""

import pytest
import torch
from torch import Tensor

from pfhedge.instruments import EuropeanOption, HestonStock, VarianceSwap
from pfhedge.nn import Hedger, MultiLayerPerceptron


class TestHedgerMultipleHedge:
    
    def test_single_vs_multiple_hedge(self):
        """Test that both single and multiple hedge instruments work correctly."""
        torch.manual_seed(42)

        def pricer(varswap) -> Tensor:
            return varswap.ul().variance

        # Create instruments
        stock = HestonStock()
        option = EuropeanOption(stock)
        varswap = VarianceSwap(stock)
        varswap.list(pricer)

        # Test single hedge (baseline)
        torch.manual_seed(42)
        model_single = MultiLayerPerceptron()
        hedger_single = Hedger(model_single, ["log_moneyness", "expiry_time", "volatility"])
        
        hedger_single.fit(option, n_paths=100, n_epochs=1)
        price_single = hedger_single.price(option, n_paths=100, n_times=3)
        assert isinstance(price_single, torch.Tensor)
        assert price_single.numel() == 1

        # Test multiple hedge 
        torch.manual_seed(42)
        model_multi = MultiLayerPerceptron(out_features=2)  # 2 outputs for 2 hedge instruments
        hedger_multi = Hedger(model_multi, ["log_moneyness", "expiry_time", "volatility"])

        # This should work without raising size mismatch errors
        hedger_multi.fit(option, hedge=[stock, varswap], n_paths=100, n_epochs=1)
        price_multi = hedger_multi.price(option, hedge=[stock, varswap], n_paths=100, n_times=3)
        assert isinstance(price_multi, torch.Tensor)
        assert price_multi.numel() == 1

    def test_hedge_tensor_shapes(self):
        """Test that hedge tensors have correct shapes for multiple instruments."""
        torch.manual_seed(42)

        def pricer(varswap) -> Tensor:
            return varswap.ul().variance

        # Create instruments
        stock = HestonStock()
        option = EuropeanOption(stock)
        varswap = VarianceSwap(stock)
        varswap.list(pricer)

        # Create hedger with multiple outputs
        model = MultiLayerPerceptron(out_features=2)
        hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"])

        # Simulate derivative
        option.simulate(n_paths=10)
        
        # Test compute_hedge shape
        hedge_ratios = hedger.compute_hedge(option, hedge=[stock, varswap])
        expected_shape = (10, 2, option.ul().spot.shape[1])  # (n_paths, n_instruments, n_timesteps)
        assert hedge_ratios.shape == expected_shape

        # Test compute_pl works without errors
        pl_value = hedger.compute_pl(option, hedge=[stock, varswap])
        assert isinstance(pl_value, torch.Tensor)
        assert pl_value.shape == (10,)  # Should return one P&L value per path

    def test_different_hedge_combinations(self):
        """Test various combinations of hedge instruments."""
        torch.manual_seed(42)

        def pricer(varswap) -> Tensor:
            return varswap.ul().variance

        # Create instruments
        stock = HestonStock()
        option = EuropeanOption(stock)
        varswap = VarianceSwap(stock)
        varswap.list(pricer)

        # Test with just stock
        model1 = MultiLayerPerceptron(out_features=1)
        hedger1 = Hedger(model1, ["log_moneyness", "expiry_time", "volatility"])
        hedger1.fit(option, hedge=[stock], n_paths=50, n_epochs=1)
        
        # Test with just variance swap
        model2 = MultiLayerPerceptron(out_features=1)
        hedger2 = Hedger(model2, ["log_moneyness", "expiry_time", "volatility"])
        hedger2.fit(option, hedge=[varswap], n_paths=50, n_epochs=1)
        
        # Test with both
        model3 = MultiLayerPerceptron(out_features=2)
        hedger3 = Hedger(model3, ["log_moneyness", "expiry_time", "volatility"])
        hedger3.fit(option, hedge=[stock, varswap], n_paths=50, n_epochs=1)
        
        # All should complete without errors
        assert True  # If we reach here, all tests passed

    def test_hedge_consistency_between_fit_and_price(self):
        """Test that hedge parameter is handled consistently between fit and price."""
        torch.manual_seed(42)

        def pricer(varswap) -> Tensor:
            return varswap.ul().variance

        # Create instruments
        stock = HestonStock()
        option = EuropeanOption(stock)
        varswap = VarianceSwap(stock)
        varswap.list(pricer)

        model = MultiLayerPerceptron(out_features=2)
        hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"])

        hedge_instruments = [stock, varswap]
        
        # Fit with specific hedge instruments
        hedger.fit(option, hedge=hedge_instruments, n_paths=100, n_epochs=1)
        
        # Price with same hedge instruments
        price1 = hedger.price(option, hedge=hedge_instruments, n_paths=100, n_times=2)
        
        # Price again to test consistency
        price2 = hedger.price(option, hedge=hedge_instruments, n_paths=100, n_times=2)
        
        # Both pricing calls should work
        assert isinstance(price1, torch.Tensor)
        assert isinstance(price2, torch.Tensor)
