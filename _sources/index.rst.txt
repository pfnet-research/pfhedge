.. PFHedge documentation master file, created by
   sphinx-quickstart on Fri Jun 11 19:12:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pfnet-research/pfhedge

PFHedge Documentation
=====================

PFHedge is a `PyTorch <https://pytorch.org>`_-based framework for `Deep Hedging <https://arxiv.org/abs/1802.03042>`_.

..
    .. image:: https://img.shields.io/pypi/pyversions/pfhedge.svg
    :target: https://pypi.org/project/pfhedge

    .. image:: https://img.shields.io/pypi/v/pfhedge.svg
    :target: https://pypi.org/project/pfhedge

    .. image:: https://github.com/pfnet-research/pfhedge/workflows/CI/badge.svg
    :target: https://github.com/pfnet-research/pfhedge/actions?query=workflow%3ACI

    .. image:: https://codecov.io/gh/pfnet-research/pfhedge/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pfnet-research/pfhedge

    .. image:: https://img.shields.io/pypi/dm/pfhedge
    :target: https://pypi.org/project/pfhedge

    .. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

API
---

.. toctree::
   :maxdepth: 1

   nn
   nn.functional
   instruments
   stochastic
   autogreek

Install
-------

.. code-block:: none

    pip install pfhedge

Getting Started
---------------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/pfnet-research/pfhedge/blob/main/examples/example_readme.ipynb

Here is a minimal example to try out Deep Hedging.

.. code-block:: python

    from pfhedge.instruments import BrownianStock
    from pfhedge.instruments import EuropeanOption
    from pfhedge.nn import Hedger
    from pfhedge.nn import MultiLayerPerceptron

    # Prepare a derivative to hedge
    deriv = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = MultiLayerPerceptron()
    hedger = Hedger(
        model, inputs=["log_moneyness", "expiry_time", "volatility", "prev_hedge"]
    )

    # Fit and price
    hedger.fit(deriv)
    price = hedger.price(deriv)

Examples
--------

More examples are provided in `GitHub repository <https://github.com/pfnet-research/pfhedge/tree/main/examples>`_.

.. toctree::
   :caption: Development
   :hidden:

   GitHub <https://github.com/pfnet-research/pfhedge>
