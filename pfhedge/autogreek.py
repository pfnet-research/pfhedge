from inspect import signature
from typing import Callable

import torch
from torch import Tensor

from ._utils.parse import parse_spot


def delta(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **kwargs
) -> Tensor:
    """Computes and returns delta of a derivative using automatic differentiation.

    Delta is a differentiation of a derivative price with respect to
    a price of underlying instrument.

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool, default=False): If `True`, graph of the derivative will be
            constructed, allowing to compute higher order derivative products.
        **kwargs: Parameters passed to `pricer`.

    Returns:
        torch.Tensor

    Examples:

        Delta of a European option can be evaluated as follows.

        >>> import pfhedge.autogreek as autogreek
        >>> from pfhedge.nn import BSEuropeanOption
        >>>
        >>> pricer = BSEuropeanOption().price
        >>> autogreek.delta(
        ...     pricer,
        ...     log_moneyness=torch.zeros(3),
        ...     expiry_time=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([0.5359, 0.5398, 0.5438])

        The result matches the analytical solution (as it should).

        >>> BSEuropeanOption().delta(
        ...     log_moneyness=torch.zeros(3),
        ...     expiry_time=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([0.5359, 0.5398, 0.5438])

        One can evaluate greeks of a price computed by a hedger.

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import WhalleyWilmott
        >>> from pfhedge.nn import Hedger
        >>>
        >>> _ = torch.manual_seed(42)
        >>>
        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4))
        >>> model = WhalleyWilmott(derivative)
        >>> hedger = Hedger(model, model.inputs())
        >>>
        >>> def pricer(spot):
        ...     return hedger.price(
        ...         derivative, init_state=(spot,), enable_grad=True
        ...     )
        >>>
        >>> autogreek.delta(pricer, spot=torch.tensor(1.0))
        tensor(0.52...)
    """
    if kwargs.get("strike") is None and kwargs.get("spot") is None:
        # Since delta does not depend on strike,
        # assign an arbitrary value (1.0) to strike if not given.
        kwargs["strike"] = torch.tensor(1.0)

    spot = parse_spot(**kwargs).requires_grad_()
    kwargs["spot"] = spot
    if "strike" in kwargs:
        kwargs["moneyness"] = spot / kwargs["strike"]
        kwargs["log_moneyness"] = (spot / kwargs["strike"]).log()

    # Delete parameters that are not in the signature of `pricer` to avoid
    # TypeError: <pricer> got an unexpected keyword argument '<parameter>'
    for parameter in list(kwargs.keys()):
        if parameter not in signature(pricer).parameters.keys():
            del kwargs[parameter]

    assert spot.requires_grad
    price = pricer(**kwargs)
    return torch.autograd.grad(
        price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]


def gamma(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **kwargs
) -> Tensor:
    """Computes and returns gamma of a derivative.

    Delta is a second-order differentiation of a derivative price with respect to
    a price of underlying instrument.

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool, default=False): If `True`, graph of the derivative will be
            constructed, allowing to compute higher order derivative products.
        **kwargs: Parameters passed to `pricer`.

    Returns:
        torch.Tensor

    Examples:

        Gamma of a European option can be evaluated as follows.

        >>> import pfhedge.autogreek as autogreek
        >>> from pfhedge.nn import BSEuropeanOption
        >>>
        >>> pricer = BSEuropeanOption().price
        >>> autogreek.gamma(
        ...     pricer,
        ...     strike=torch.ones(3),
        ...     log_moneyness=torch.zeros(3),
        ...     expiry_time=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([2.2074, 1.9848, 1.8024])

        The result matches the analytical solution (as it should).

        >>> BSEuropeanOption().gamma(
        ...     log_moneyness=torch.zeros(3),
        ...     expiry_time=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([2.2074, 1.9848, 1.8024])
    """
    spot = parse_spot(**kwargs).requires_grad_()
    kwargs["spot"] = spot
    if "strike" in kwargs:
        kwargs["moneyness"] = spot / kwargs["strike"]
        kwargs["log_moneyness"] = (spot / kwargs["strike"]).log()

    tensor_delta = delta(pricer, create_graph=True, **kwargs).requires_grad_()
    return torch.autograd.grad(
        tensor_delta,
        inputs=spot,
        grad_outputs=torch.ones_like(tensor_delta),
        create_graph=create_graph,
    )[0]
