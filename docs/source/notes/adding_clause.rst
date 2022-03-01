Adding clause to derivative
===========================

One can customize a derivative by registering additional clauses;
see :func:`pfhedge.instruments.BaseDerivative.add_clause`.

Let us see how to add clauses to derivatives by taking European capped call option as an example.
This option is a variant of a European option, which is given by

.. code:: python

    >>> from pfhedge.instruments import BrownianStock
    >>> from pfhedge.instruments import EuropeanOption
    ...
    >>> strike = 1.0
    >>> maturity = 1.0
    >>> stock = BrownianStock()
    >>> european = EuropeanOption(stock, strike=strike, maturity=maturity)

The capped call associates ''a barrier clause'';
if the underlier's spot reaches the barrier price :math:`B`,
being greater than the strike :math:`K` and the spot at inception,
the option immediately expires and pays off its intrinsic value at that moment :math:`B - K`.

This clause can be registered to a derivative as follows:

.. code:: python

    >>> def cap_clause(derivative, payoff):
    ...     barrier = 1.4
    ...     max_spot = derivative.ul().spot.max(-1).values
    ...     capped_payoff = torch.full_like(payoff, barrier - strike)
    ...     return torch.where(max_spot < barrier, payoff, capped_payoff)
    ...
    >>> capped_european = EuropeanOption(stock, strike=strike, maturity=maturity)
    >>> capped_european.add_clause("cap_clause", cap_clause)

The method ``add_clause`` adds the clause and its name to the derivative.
Here the function ``cap_caluse`` represents the clause to modify the payoff depending on the state of the derivative.
The clause function should have the signature ``clause(derivative, payoff) -> modified payoff``.

The payoff would be capped as intended:

.. code:: python

    >>> n_paths = 100000
    >>> capped_european.simulate(n_paths=n_paths)
    >>> european.payoff().max()
    >>> # 1.2...
    >>> capped_european.payoff().max()
    >>> # 0.4

The price of the capped European call option can be evaluated by using a European option as a control variates.

.. code:: python

    >>> from math import sqrt
    ...
    >>> payoff_european = european.payoff()
    >>> payoff_capped_european = capped_european.payoff()
    >>> bs_price = BlackScholes(european).price(0.0, european.maturity, stock.sigma).item()
    >>> price = bs_price + (payoff_capped_european - payoff_european).mean().item()
    >>> error = (payoff_capped_european - payoff_european).std().item() / sqrt(n_paths)
    >>> bs_price
    >>> # 0.07967...
    >>> price
    >>> # 0.07903...
    >>> error
    >>> # 0.00012...
