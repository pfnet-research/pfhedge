#为期权添加条款
import sys

sys.path.append("..")

from math import sqrt

import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes


def main():
    torch.manual_seed(42) #随机数种子

    strike = 1.0
    maturity = 1.0
    stock = BrownianStock() #生成随机价格路径
    european = EuropeanOption(stock, strike=strike, maturity=maturity)

    #上限条款，当标的资产价格超过1.4时，期权收益限制在0.4
    def cap_clause(derivative, payoff):
        barrier = 1.4
        max_spot = derivative.ul().spot.max(-1).values
        capped_payoff = torch.full_like(payoff, barrier - strike)
        return torch.where(max_spot < barrier, payoff, capped_payoff)

    capped_european = EuropeanOption(stock, strike=strike, maturity=maturity)
    capped_european.add_clause("cap_clause", cap_clause)

    n_paths = 100000
    capped_european.simulate(n_paths=n_paths) #蒙特卡洛模拟

    #每条模拟路径的期权收益向量————取平均后用于估计期权价格
    payoff_european = european.payoff()
    payoff_capped_european = capped_european.payoff()

    #最大期权收益值————表征风险特征
    max_spot = payoff_european.max().item()
    capped_max_spot = payoff_capped_european.max().item()
    print("Max payoff of vanilla European:", max_spot)
    print("Max payoff of capped  European:", capped_max_spot)

    #控制变量，方差减少技术
    #价格 = BS价格 + E[带上限收益-普通收益]
    #error1 < error0
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
