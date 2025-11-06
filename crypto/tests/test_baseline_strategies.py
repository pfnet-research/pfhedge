import pytest
import torch
import numpy as np

from crypto.instruments import BitcoinPerpetualBrownian, BitcoinEuropeanOption
from crypto.strategies import calculate_bs_hedge_pnl


class TestBlackScholesBaseline:

    def test_bs_delta_shape(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=10, init_state=(1.0,))

        bs_delta = option.black_scholes_delta()

        assert bs_delta.shape == option.underlier.spot.shape
        assert not torch.any(torch.isnan(bs_delta))
        assert not torch.any(torch.isinf(bs_delta))

    def test_bs_delta_bounds(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=10, init_state=(1.0,))

        bs_delta = option.black_scholes_delta()

        # Call delta should be in [0, 1]
        assert torch.all(bs_delta >= 0), "Call delta should be >= 0"
        assert torch.all(bs_delta <= 1), "Call delta should be <= 1"

    def test_bs_pnl_calculation(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=10, init_state=(1.0,))

        spots = option.underlier.spot
        bs_delta = option.black_scholes_delta()
        payoffs = option.payoff()

        pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=0.0005,
        )

        assert pnl.shape == spots.shape
        assert not torch.any(torch.isnan(pnl))
        assert not torch.any(torch.isinf(pnl))

    def test_bs_pnl_with_zero_cost(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=10, init_state=(1.0,))

        spots = option.underlier.spot
        bs_delta = option.black_scholes_delta()
        payoffs = option.payoff()

        pnl_no_cost = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=0.0,
        )

        pnl_with_cost = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=0.0005,
        )

        # PnL with cost should be lower (more negative or less positive)
        assert torch.all(
            pnl_with_cost[:, -1] <= pnl_no_cost[:, -1] + 1e-6
        ), "PnL with transaction costs should be lower"

    def test_bs_pnl_final_value_reasonable(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=100, init_state=(1.0,))

        spots = option.underlier.spot
        bs_delta = option.black_scholes_delta()
        payoffs = option.payoff()

        pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=0.0005,
        )

        final_pnl = pnl[:, -1]

        # With good hedging, PnL should be close to zero (small profit/loss)
        # Allow range of [-0.5, 0.5] for 100 paths
        assert final_pnl.mean().abs() < 0.1, "Mean PnL should be close to zero"
        assert final_pnl.std() < 0.3, "PnL std should be relatively small"


class TestNoHedgeBaseline:

    def test_no_hedge_pnl(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=10, init_state=(1.0,))

        spots = option.underlier.spot
        payoffs = option.payoff()

        # No-hedge: delta always zero
        no_hedge_delta = torch.zeros_like(spots)

        pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=no_hedge_delta,
            payoffs=payoffs,
            cost=0.0,
        )

        # No-hedge PnL should just be -payoff (we sold option, didn't hedge)
        final_pnl = pnl[:, -1]
        expected_pnl = -payoffs

        assert torch.allclose(
            final_pnl, expected_pnl, rtol=1e-5
        ), "No-hedge PnL should equal -payoff"

    def test_no_hedge_higher_variance(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=200, init_state=(1.0,))

        spots = option.underlier.spot
        bs_delta = option.black_scholes_delta()
        payoffs = option.payoff()

        # BS hedge PnL
        bs_pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=0.0,
        )

        # No-hedge PnL
        no_hedge_delta = torch.zeros_like(spots)
        no_hedge_pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=no_hedge_delta,
            payoffs=payoffs,
            cost=0.0,
        )

        bs_std = bs_pnl[:, -1].std()
        no_hedge_std = no_hedge_pnl[:, -1].std()

        # No-hedge should have significantly higher variance
        assert (
            no_hedge_std > bs_std * 2
        ), f"No-hedge std ({no_hedge_std:.4f}) should be much higher than BS std ({bs_std:.4f})"


class TestUnhedgedBaseline:

    def test_unhedged_pnl_equals_payoff(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=100, init_state=(1.0,))

        payoffs = option.payoff()

        # Unhedged baseline: just hold the option and collect payoff
        # No hedging activity, so PnL is just the payoff distribution
        unhedged_pnl = payoffs

        # Verify it's reasonable
        assert not torch.any(torch.isnan(unhedged_pnl))
        assert not torch.any(torch.isinf(unhedged_pnl))

        # For ATM call option, payoff should be >= 0
        assert torch.all(unhedged_pnl >= 0), "Call option payoffs must be non-negative"

        # Some paths should be ITM (positive payoff)
        assert torch.any(unhedged_pnl > 0), "Some paths should finish in-the-money"


class TestBaselineComparison:

    def test_all_baselines_computable(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=50, init_state=(1.0,))

        spots = option.underlier.spot
        bs_delta = option.black_scholes_delta()
        payoffs = option.payoff()

        # 1. BS hedge
        bs_pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=0.0005,
        )

        # 2. No-hedge
        no_hedge_delta = torch.zeros_like(spots)
        no_hedge_pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=no_hedge_delta,
            payoffs=payoffs,
            cost=0.0,
        )

        # 3. Unhedged (just payoffs)
        unhedged_pnl = payoffs

        # All should be valid
        assert not torch.any(torch.isnan(bs_pnl))
        assert not torch.any(torch.isnan(no_hedge_pnl))
        assert not torch.any(torch.isnan(unhedged_pnl))

        # BS should have lowest variance
        bs_std = bs_pnl[:, -1].std()
        no_hedge_std = no_hedge_pnl[:, -1].std()
        unhedged_std = unhedged_pnl.std()

        assert (
            bs_std < no_hedge_std
        ), "BS hedge should have lower variance than no-hedge"
        assert (
            bs_std < unhedged_std
        ), "BS hedge should have lower variance than unhedged"

    def test_variance_ranking(self):
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
        option = BitcoinEuropeanOption(
            underlier=underlier, strike=1.0, maturity=14 / 365, call=True
        )
        option.simulate(n_paths=200, init_state=(1.0,))

        spots = option.underlier.spot
        bs_delta = option.black_scholes_delta()
        payoffs = option.payoff()

        # Compute all baselines
        bs_pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, cost=0.0)
        no_hedge_pnl = calculate_bs_hedge_pnl(
            spots, torch.zeros_like(spots), payoffs, cost=0.0
        )
        unhedged_pnl = payoffs

        # Variance ranking: BS < unhedged < no-hedge
        bs_var = bs_pnl[:, -1].var()
        no_hedge_var = no_hedge_pnl[:, -1].var()
        unhedged_var = unhedged_pnl.var()

        print(f"\nVariance comparison:")
        print(f"  BS hedge: {bs_var:.6f}")
        print(f"  No-hedge: {no_hedge_var:.6f}")
        print(f"  Unhedged: {unhedged_var:.6f}")

        assert bs_var < unhedged_var, "BS should have lower variance than unhedged"
        assert bs_var < no_hedge_var, "BS should have lower variance than no-hedge"


def test_baseline_summary():
    torch.manual_seed(42)

    underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=0.8)
    option = BitcoinEuropeanOption(
        underlier=underlier, strike=1.0, maturity=14 / 365, call=True
    )
    option.simulate(n_paths=100, init_state=(1.0,))

    # Verify all baselines work
    spots = option.underlier.spot
    bs_delta = option.black_scholes_delta()
    payoffs = option.payoff()

    # BS hedge
    bs_pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, cost=0.0005)
    assert not torch.any(torch.isnan(bs_pnl))

    # No-hedge
    no_hedge_pnl = calculate_bs_hedge_pnl(
        spots, torch.zeros_like(spots), payoffs, cost=0.0
    )
    assert not torch.any(torch.isnan(no_hedge_pnl))

    # Unhedged
    unhedged_pnl = payoffs
    assert not torch.any(torch.isnan(unhedged_pnl))

    print("\n✅ All baseline strategies verified!")
    print(f"   BS hedge std: {bs_pnl[:, -1].std():.4f}")
    print(f"   No-hedge std: {no_hedge_pnl[:, -1].std():.4f}")
    print(f"   Unhedged std: {unhedged_pnl.std():.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
