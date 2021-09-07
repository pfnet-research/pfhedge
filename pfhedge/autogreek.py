from inspect import signature
from typing import Callable

import torch
from torch import Tensor

from ._utils.parse import parse_spot
from ._utils.parse import parse_volatility


def delta(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **params
) -> Tensor:
    """Computes and returns delta of a derivative using automatic differentiation.

    Delta is a differentiation of a derivative price with respect to
    a price of underlying instrument.

    Note:
        The keyword argument ``**params`` should contain at least one of
        the following combinations:

        - ``spot``
        - ``moneyness`` and ``strike``
        - ``log_moneyness`` and ``strike``

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool, default=False): If ``True``,
            graph of the derivative will be constructed,
            allowing to compute higher order derivative products.
        **params: Parameters passed to ``pricer``.

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
        ...     time_to_maturity=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([0.5359, 0.5398, 0.5438])

        The result matches the analytical solution (as it should).

        >>> BSEuropeanOption().delta(
        ...     log_moneyness=torch.zeros(3),
        ...     time_to_maturity=torch.ones(3),
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
        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4)).to(torch.float64)
        >>> model = WhalleyWilmott(derivative)
        >>> hedger = Hedger(model, model.inputs()).to(torch.float64)
        >>>
        >>> def pricer(spot):
        ...     return hedger.price(
        ...         derivative, init_state=(spot,), enable_grad=True
        ...     )
        >>>
        >>> autogreek.delta(pricer, spot=torch.tensor(1.0))
        tensor(0.5...)
    """
    if params.get("strike") is None and params.get("spot") is None:
        # Since delta does not depend on strike,
        # assign an arbitrary value (1.0) to strike if not given.
        params["strike"] = torch.tensor(1.0)

    spot = parse_spot(**params).requires_grad_()
    params["spot"] = spot
    if "strike" in params:
        params["moneyness"] = spot / params["strike"]
        params["log_moneyness"] = (spot / params["strike"]).log()

    # Delete parameters that are not in the signature of pricer to avoid
    # TypeError: <pricer> got an unexpected keyword argument '<parameter>'
    for parameter in list(params.keys()):
        if parameter not in signature(pricer).parameters.keys():
            del params[parameter]

    price = pricer(**params)
    return torch.autograd.grad(
        price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]


def gamma(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **params
) -> Tensor:
    """Computes and returns gamma of a derivative.

    Delta is a second-order differentiation of a derivative price with respect to
    a price of underlying instrument.

    Note:
        The keyword argument ``**params`` should contain at least one of
        the following combinations:

        - ``spot``
        - ``moneyness`` and ``strike``
        - ``log_moneyness`` and ``strike``

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool, default=False): If ``True``,
            graph of the derivative will be constructed,
            allowing to compute higher order derivative products.
        **params: Parameters passed to ``pricer``.

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
        ...     time_to_maturity=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([2.2074, 1.9848, 1.8024])

        The result matches the analytical solution (as it should).

        >>> BSEuropeanOption().gamma(
        ...     log_moneyness=torch.zeros(3),
        ...     time_to_maturity=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([2.2074, 1.9848, 1.8024])
    """
    spot = parse_spot(**params).requires_grad_()
    params["spot"] = spot
    if "strike" in params:
        params["moneyness"] = spot / params["strike"]
        params["log_moneyness"] = (spot / params["strike"]).log()

    tensor_delta = delta(pricer, create_graph=True, **params).requires_grad_()
    return torch.autograd.grad(
        tensor_delta,
        inputs=spot,
        grad_outputs=torch.ones_like(tensor_delta),
        create_graph=create_graph,
    )[0]


def vega(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **params
) -> Tensor:
    """Computes and returns vega of a derivative using automatic differentiation.

    Vega is a differentiation of a derivative price with respect to
    a variance of underlying instrument.

    Note:
        The keyword argument ``**params`` should contain at least one of the
        following parameters:

        - ``volatility``
        - ``variance``

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool, default=False): If ``True``,
            graph of the derivative will be constructed,
            allowing to compute higher order derivative products.
        **params: Parameters passed to ``pricer``.

    Returns:
        torch.Tensor

    Examples:

        Vega of a European option can be evaluated as follows.

        >>> import pfhedge.autogreek as autogreek
        >>> from pfhedge.nn import BSEuropeanOption
        >>>
        >>> pricer = BSEuropeanOption().price
        >>> autogreek.vega(
        ...     pricer,
        ...     log_moneyness=torch.zeros(3),
        ...     time_to_maturity=torch.ones(3),
        ...     volatility=torch.tensor([0.18, 0.20, 0.22]),
        ... )
        tensor([0.3973, 0.3970, 0.3965])
    """
    volatility = parse_volatility(**params).requires_grad_()
    params["volatility"] = volatility
    params["variance"] = volatility.square()

    # Delete parameters that are not in the signature of pricer to avoid
    # TypeError: <pricer> got an unexpected keyword argument '<parameter>'
    for parameter in list(params.keys()):
        if parameter not in signature(pricer).parameters.keys():
            del params[parameter]

    price = pricer(**params)
    return torch.autograd.grad(
        price,
        inputs=volatility,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]
