#利用期权组合对冲方差互换
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes

if __name__ == "__main__":
    #创建一系列看涨+看跌欧式期权
    options_list = []
    strikes_list = []
    for call in (True, False):
        for strike in torch.arange(70, 180, 10):
            option = EuropeanOption(BrownianStock(), call=call, strike=strike.item())
            options_list.append(option)
            strikes_list.append(strike)

    #计算Vega暴露
    #Vega：期权价格对波动率的偏导数(敏感度)
    spot = torch.linspace(50, 200, 100) #标的资产价格范围
    t = options_list[0].maturity #到期时间
    v = options_list[0].ul().sigma #波动率

    plt.figure()
    total_vega = torch.zeros_like(spot)
    for option, strike in zip(options_list, strikes_list):
        lm = (spot / strike).log()
        vega = BlackScholes(option).vega(lm, t, v) / (strike ** 2)
        total_vega += vega
        if option.call:
            # 2 is for call and put
            plt.plot(spot.numpy(), 2 * vega.numpy())
    plt.plot(spot.numpy(), total_vega.numpy(), color="k", lw=2)
    plt.savefig("./output/options-vega.png")
