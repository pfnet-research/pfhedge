<h1 align="center">PFHedge: Deep Hedging in PyTorch</h1>

[![python](https://img.shields.io/pypi/pyversions/pfhedge.svg)](https://pypi.org/project/pfhedge)
[![pypi](https://img.shields.io/pypi/v/pfhedge.svg)](https://pypi.org/project/pfhedge)
[![CI](https://github.com/pfnet-research/pfhedge/workflows/CI/badge.svg)](https://github.com/pfnet-research/pfhedge/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/pfnet-research/pfhedge/branch/master/graph/badge.svg)](https://codecov.io/gh/pfnet-research/pfhedge)
[![downloads](https://img.shields.io/pypi/dm/pfhedge)](https://pypi.org/project/pfhedge)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation](https://simaki.github.io/pfhedge/)

**PFHedge** is a [PyTorch](https://pytorch.org/)-based framework for [Deep Hedging][deep-hedging-arxiv].

## What is Deep Hedging?

[**Deep Hedging**][deep-hedging-arxiv] is a deep learning-based framework to hedge financial derivatives.

Hedging [financial derivatives](https://en.wikipedia.org/wiki/Derivative_(finance)) in the presence of market frictions (e.g., transaction cost) is a challenging task.
In the absence of market frictions, the perfect hedge is accessible based on the [Black-Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model).
The real market, in contrast, always involves frictions and thereby makes hedging optimization much more challenging.
Since the analytic formulas (such as the [Black-Scholes formula](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula)) are no longer available in such a market, human traders may manually adjust model-based Greeks to hedge and price derivatives based on their experiences.

[Deep Hedging][deep-hedging-arxiv] is a ground-breaking framework to automate and optimize such hedging operations.
In this framework, a neural network is trained to hedge derivatives so that it minimizes a proper [risk measure](https://en.wikipedia.org/wiki/Risk_measure).
By virtue of the high representability of a neural network and modern optimization algorithms, one can expect to achieve the optimal hedge by training a neural network.
Indeed, the experiments in [Bühler *et al.* 18][deep-hedging-qf] and [Imaki *et al.* 21][ntb-network-arxiv] show high feasibility and scalability of Deep Hedging algorithms for options under transaction costs.

Global investment banks are looking to [replace the Greeks-based manual hedging](https://www.risk.net/derivatives/6875321/deep-hedging-and-the-end-of-the-black-scholes-era) with Deep Hedging and [slash up to 80% of hedging costs](https://www.risk.net/derivatives/6691696/jp-morgan-turns-to-machine-learning-for-options-hedging).
This could be the "game-changer" in the trillion-dollar industry of derivatives.

PFHedge enables you to experience this revolutionary framework on your own.
You can try, tweak, and delve into Deep Hedging algorithms using PyTorch.
We hope PFHedge accelerates the research and development of Deep Hedging.

## Features

### Imperative Experiences

* PFHedge is designed to be intuitive and imperative to streamline your research on Deep Hedging.
* You can quickly build a `Hedger` and then `fit` and `price` derivatives right away.
* You can easily tweak your model, risk measure, derivative, optimizer, and other setups on the fly.

### Seamless Integration with [PyTorch](https://pytorch.org/)

* PFHedge is built to be deeply integrated into [PyTorch](https://pytorch.org/).
* Your Deep-Hedger can be built as a [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) and trained by any [`Optimizer`](https://pytorch.org/docs/stable/optim.html).
* You can use GPUs to boost your hedging optimization (See below).

### Effortless Extensions

* You can build new hedging models, derivatives, and features with little glue code.
* You can build new hedging models by just subclassing [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
* You can quickly try out your own stochastic processes, derivatives, and input features.

### Batteries Included

* PFHedge provides useful [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)s for Deep Hedging in [`pfhedge.nn`](https://github.com/pfnet-research/pfhedge/tree/main/pfhedge/nn).
* You can create [Black-Scholes' delta-hedging](https://en.wikipedia.org/wiki/Delta_neutral), [Whalley-Wilmott's strategy][whalley-wilmott], and so forth.
* Common risk measures such as [an entropic risk measure](https://en.wikipedia.org/wiki/Entropic_risk_measure) and [an expected shortfall](https://en.wikipedia.org/wiki/Expected_shortfall) are available.

## Install

```sh
$ pip install pfhedge
```

## How to Use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][example-readme-colab]

### Prepare a Derivative to Hedge

Financial instruments can be classified into two types:

* **`Primary` instruments**: A primary instrument is a basic financial instrument that is traded on a market, and therefore their prices are accessible as the market prices. Examples include stocks, bonds, commodities, and currencies.
* **`Derivative` instruments**: A derivative is a financial instrument whose payoff is contingent on a primary instrument. An (over-the-counter) derivative is not traded on the market, and therefore the price is not directly accessible. Examples include [`EuropeanOption`](https://en.wikipedia.org/wiki/Option_style#American_and_European_options), [`LookbackOption`](https://en.wikipedia.org/wiki/Lookback_option), and so forth.

We consider a `BrownianStock`, which is a stock following the [geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion), and a [`EuropeanOption`](https://en.wikipedia.org/wiki/Option_style#American_and_European_options) which is contingent on it.
We assume that the stock has a transaction cost of 1 basis point.

```py
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption

stock = BrownianStock(cost=1e-4)
deriv = EuropeanOption(stock)
```

### Create Your Hedger

A `Hedger` in Deep Hedging is basically characterized by three elements:

* **Features**: A hedger uses any market information as input features.
    - `'log_moneyness'`: [Log-moneyness](https://en.wikipedia.org/wiki/Moneyness) of the stock.
    - `'expiry_time'`: Remaining time to the [maturity](https://en.wikipedia.org/wiki/Maturity_(finance)) of the option.
    - `'volatility'`: [Volatility](https://en.wikipedia.org/wiki/Volatility_(finance)) of the stock.
    - `'prev_hedge'`: The hedge ratio at the previous time step.
* **Model**: A hedger's model computes the hedge ratio at the next time step from input features.
    - `MultiLayerPerceptron`: [Multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).
    - `BlackScholes`: [Black-Scholes](https://en.wikipedia.org/wiki/Delta_neutral)' delta-hedging strategy.
    - `WhalleyWilmott`: [Whalley-Wilmott][whalley-wilmott]'s asymptotically optimal strategy for small costs.
    - Any PyTorch [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) which you build.
* **Criterion**: A hedger wishes to minimize their [risk measure](https://en.wikipedia.org/wiki/Risk_measure).
    - `EntropicRiskMeasure`: [Entropic Risk Measure](https://en.wikipedia.org/wiki/Entropic_risk_measure), a risk measure derived from [exponential utility](https://en.wikipedia.org/wiki/Exponential_utility).
    - `ExpectedShortFall`: [Expected Shortfall](https://en.wikipedia.org/wiki/Expected_shortfall) or CVaR, a common measure to assess portfolio risk.

We here use a multi-layer perceptron as our model.

```py
from pfhedge import Hedger
from pfhedge.nn import MultiLayerPerceptron

model = MultiLayerPerceptron()
hedger = Hedger(model, features=["log_moneyness", "expiry_time", "volatility", "prev_hedge"])
```

The `hedger` is also a [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

```py
hedger
# Hedger(
#   features=['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge'],
#   (model): MultiLayerPerceptron(
#     (0): LazyLinear(in_features=None, out_features=32, bias=True)
#     (1): ReLU()
#     (2): LazyLinear(in_features=None, out_features=32, bias=True)
#     (3): ReLU()
#     (4): LazyLinear(in_features=None, out_features=32, bias=True)
#     (5): ReLU()
#     (6): LazyLinear(in_features=None, out_features=32, bias=True)
#     (7): ReLU()
#     (8): LazyLinear(in_features=None, out_features=1, bias=True)
#     (9): Identity()
#   )
#   (criterion): EntropicRiskMeasure()
# )
```

### Fit and Price

Now we train our `hedger` so that it minimizes the risk measure through hedging.

The `hedger` is trained as follows.
In each epoch, we generate Monte Carlo paths of the asset prices and let the `hedger` hedge the derivative by trading the stock.
The hedger's risk measure (`EntropicRiskMeasure()` in our case) is computed from the resulting profit and loss distribution, and the parameters in the `model` are updated.

```py
hedger.fit(deriv, n_epochs=200, n_paths=10000)
```

Once we have trained the `hedger`, we can evaluate the derivative price as utility indifference price (For details, see [Deep Hedging][deep-hedging-arxiv] and references therein).

```py
price = hedger.price(deriv)
```

## More Examples

### Use GPU

To employ the desired [`device`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device) (and/or [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype)) in fitting and pricing, use `to` method.

```py
device = torch.device("cuda:0")

deriv = EuropeanOption(BrownianStock(cost=1e-4))
deriv.to(device)

hedger = Hedger(...)
hedger.to(device)
```

### Black-Scholes' Delta-Hedging Strategy

In this strategy, a hedger incessantly rebalances their portfolio and keeps it [delta-neutral](https://en.wikipedia.org/wiki/Delta_neutral).
The hedge-ratio at each time step is given by the Black-Scholes' delta.

This strategy is the optimal one in the absence of cost.
On the other hand, this strategy transacts too frequently and consumes too much transaction cost.

```py
from pfhedge import Hedger
from pfhedge.nn import BlackScholes

deriv = EuropeanOption(BrownianStock(cost=1e-4))

model = BlackScholes(deriv)
hedger = Hedger(model, model.features())
```

### Whalley-Wilmott's Asymptotically Optimal Strategy for Small Costs

This strategy is proposed by [Whalley *et al.* 1997][whalley-wilmott] and is proved to be optimal for asymptotically small transaction costs.

In this strategy, a hedger always maintains their hedge ratio in the range (called no-transaction band) while they never transact inside this range.
This strategy is supposed to be optimal in the limit of small transaction costs, while suboptimal for large transaction costs.

```py
from pfhedge import Hedger
from pfhedge.nn import WhalleyWilmott

deriv = EuropeanOption(BrownianStock(cost=1e-4))

model = WhalleyWilmott(deriv)
hedger = Hedger(model, model.features())
```

### Your Own Module

You can employ any [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) you build as a hedging model.
The input/output shapes is `(N, H_in) -> (N, 1)`, where `N` is the number of Monte Carlo paths of assets and `H_in` is the number of input features.

Here we show an example of **No-Transaction Band Network**, which is proposed in [Imaki *et al.* 21][ntb-network-arxiv].

```py
import torch
import torch.nn.functional as fn
from pfhedge.nn import BlackScholes
from pfhedge.nn import Clamp
from pfhedge.nn import MultiLayerPerceptron


class NoTransactionBandNet(torch.nn.Module):
    def __init__(self, liability):
        super().__init__()

        self.delta = BlackScholes(liability)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def features(self):
        return self.delta.features() + ["prev_hedge"]

    def forward(self, x):
        prev_hedge = x[:, [-1]]

        delta = self.delta(x[:, :-1])
        width = self.mlp(x[:, :-1])

        lower = delta - fn.leaky_relu(width[:, [0]])
        upper = delta + fn.leaky_relu(width[:, [1]])

        return self.clamp(prev_hedge, min_value=lower, max_value=upper)


model = NoTransactionBandNet()
hedger = Hedger(model, model.features())
```

## Contribution

Any contributions to PFHedge are more than welcome!

* GitHub Issues: Bug reports, feature requests, and questions.
* Pull Requests: Bug-fixes, feature implementations, and documentation updates.

This project is owned by [Preferred Networks](https://www.preferred.jp/en/) and maintained by [Shota Imaki](https://github.com/simaki).

## References

* Hans Bühler, Lukas Gonon, Josef Teichmann and Ben Wood, "[Deep hedging][deep-hedging-qf]". Quantitative Finance, 2019, 19, 1271-1291. arXiv:[1609.05213][deep-hedging-arxiv] \[q-fin.CP\].
* Hans Bühler, Lukas Gonon, Josef Teichmann, Ben Wood, Baranidharan Mohan and Jonathan Kochems, [Deep Hedging: Hedging Derivatives Under Generic Market Frictions Using Reinforcement Learning][deep-hedging-wp] (March 19, 2019). Swiss Finance Institute Research Paper No. 19-80.
* Shota Imaki, Kentaro Imajo, Katsuya Ito, Kentaro Minami and Kei Nakagawa, "No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging". arXiv:[2103.01775][ntb-network-arxiv] \[q-fin.CP\].

[deep-hedging-arxiv]: https://arxiv.org/abs/1802.03042
[deep-hedging-qf]: https://www.tandfonline.com/doi/abs/10.1080/14697688.2019.1571683
[deep-hedging-wp]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3355706
[ntb-network-arxiv]: https://arxiv.org/abs/2103.01775
[whalley-wilmott]: https://doi.org/10.1111/1467-9965.00034
[NoTransactionBandNetwork]: https://github.com/pfnet-research/NoTransactionBandNetwork
[CONTRIBUTING.md]: https://github.com/pfnet-research/pfhedge/blob/master/CONTRIBUTING.md
[example-readme-colab]: https://colab.research.google.com/github/pfnet-research/pfhedge/blob/main/examples/example_readme.ipynb
