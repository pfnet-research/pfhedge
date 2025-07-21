# Example to add a knock-out clause to a derivative

import sys

sys.path.append("..")

if __name__ == "__main__":
    import torch
    from torch import Tensor

    from pfhedge.instruments import BaseDerivative
    from pfhedge.instruments import BrownianStock
    from pfhedge.instruments import EuropeanOption

    derivative = EuropeanOption(BrownianStock(cost=1e-4))

    def knockout(derivative: BaseDerivative, payoff: Tensor) -> Tensor:
        max = derivative.ul().spot.max(-1).values
        return payoff.where(max <= 1.1, torch.zeros_like(max))

    torch.manual_seed(42)
    derivative.simulate(n_paths=1000)

    payoff = derivative.payoff()
    print("Max payoff without a knockout clause:", payoff.max().item())

    derivative.add_clause("knockout", knockout)

    payoff = derivative.payoff()
    print("Max payoff with a knockout clause:", payoff.max().item())
