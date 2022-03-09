.. role:: hidden
    :class: hidden-section

pfhedge.nn.functional
=====================

.. currentmodule:: pfhedge.nn.functional

Payoff Functions
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated

    european_payoff
    lookback_payoff
    american_binary_payoff
    european_binary_payoff
    european_forward_start_payoff

Nonlinear activation functions
------------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated

    leaky_clamp
    clamp

Criterion Functions
-------------------

.. autosummary::
    :nosignatures:
    :toctree: generated

    exp_utility
    isoelastic_utility
    entropic_risk_measure
    expected_shortfall
    value_at_risk

Black-Scholes formulas
----------------------

.. autosummary::
    :nosignatures:
    :toctree: generated

    bs_european_price
    bs_european_delta
    bs_european_gamma
    bs_european_vega
    bs_european_theta
    bs_american_binary_price
    bs_american_binary_delta
    bs_american_binary_gamma
    bs_american_binary_vega
    bs_american_binary_theta
    bs_european_binary_price
    bs_european_binary_delta
    bs_european_binary_gamma
    bs_european_binary_vega
    bs_european_binary_theta
    bs_lookback_price
    bs_lookback_delta
    bs_lookback_gamma
    bs_lookback_vega
    bs_lookback_theta

Other Functions
---------------

.. autosummary::
    :nosignatures:
    :toctree: generated

    topp
    realized_variance
    realized_volatility
    terminal_value
    ncdf
    npdf
    d1
    d2
    svi_variance
