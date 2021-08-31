Getting Started
===============

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/pfnet-research/pfhedge/blob/main/examples/example_readme.ipynb

Here is a minimal example to try out Deep Hedging.

.. code-block:: python

    import torch
    from pfhedge.instruments import BrownianStock
    from pfhedge.instruments import EuropeanOption
    from pfhedge.nn import Hedger
    from pfhedge.nn import MultiLayerPerceptron

    torch.manual_seed(42)

    # Prepare a derivative to hedge
    deriv = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = MultiLayerPerceptron()
    hedger = Hedger(
        model, inputs=["log_moneyness", "time_to_maturity", "volatility", "prev_hedge"]
    )

    # Fit and price
    hedger.fit(deriv)
    price = hedger.price(deriv)

    print("Price:", price.item())
