import sys

sys.path.append("..")


def main():
    import torch

    torch.manual_seed(42)

    from math import sqrt

    from pfhedge.instruments import BrownianStock
    from pfhedge.instruments import EuropeanOption
    from pfhedge.nn import BlackScholes

    strike = 1.0
    maturity = 1.0
    stock = BrownianStock()
    european = EuropeanOption(stock, strike=strike, maturity=maturity)

    def cap_clause(derivative, payoff):
        barrier = 1.4
        max = derivative.ul().spot.max(-1).values
        capped_payoff = torch.full_like(payoff, barrier - strike)
        return torch.where(max < barrier, payoff, capped_payoff)

    capped_european = EuropeanOption(stock, strike=strike, maturity=maturity)
    capped_european.add_clause("cap_clause", cap_clause)

    n_paths = 100000
    capped_european.simulate(n_paths=n_paths)

    payoff_european = european.payoff()
    payoff_capped_european = capped_european.payoff()

    max = payoff_european.max().item()
    capped_max = payoff_capped_european.max().item()
    print("Max payoff of vanilla European:", max)
    print("Max payoff of capped  European:", capped_max)

    # Price using control variates
    bs_price = BlackScholes(european).price(0.0, european.maturity, stock.sigma).item()
    value0 = payoff_capped_european.mean().item()
    value1 = bs_price + (payoff_capped_european - payoff_european).mean().item()
    error0 = payoff_capped_european.std().item() / sqrt(n_paths)
    error1 = (payoff_capped_european - payoff_european).std().item() / sqrt(n_paths)

    print("BS price of vanilla European:", bs_price)
    print("Price of capped European without control variates:", value0)
    print("Price of capped European with    control variates:", value1)
    print("Error of capped European without control variates:", error0)
    print("Error of capped European with    control variates:", error1)


if __name__ == "__main__":
    main()
