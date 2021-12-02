.. currentmodule:: pfhedge.instruments

.. _instrument-attributes-doc:

Instrument dtype and device
===========================

The attributes ``dtype`` and ``device`` of a primary instrument specify
the dtype and devices with which the buffers of the instrument are represented.
The attributes ``dtype`` and ``device`` of a derivative instrument
are aliased to the dtype/device of its underlier.

A instrument of specific dtype/device can be constructed by
passing a ``torch.dtype`` and/or a ``torch.device`` to a constructor.

.. code:: python

    >>> from pfhedge.instruments import BrownianStock
    >>>
    >>> stock = BrownianStock(dtype=torch.float64)
    >>> stock.simulate(n_paths=2, time_horizon=2 / 250)
    >>> stock.spot
    tensor([[..., ..., ...],
            [..., ..., ...]], dtype=torch.float64)
    >>> stock = BrownianStock(device="cuda:0", dtype=torch.float64)
    >>> stock.simulate(n_paths=2, time_horizon=2 / 250)
    >>> stock.spot
    tensor([[..., ..., ...],
            [..., ..., ...]], device='cuda:0')

The methods :meth:`BasePrimary.to()` and :meth:`BaseDerivative.to()` work as
:meth:`torch.nn.Module.to` and cast/move the dtype and device of the instrument
to the desired ones.

In the rest of this note we will describe these attributes in detail
and show how to use ``to`` method.

Instrument.dtype
----------------

Primary
^^^^^^^

One can simulate the movement of a primary instrument (e.g., :class:`EuropeanOption`)
to get the associated buffers (e.g., ``spot``).

The simulation will be performed with a :class:`torch.dtype`
specified by :attr:`BasePrimary.dtype`.
The default dtype is the global default (see :func:`torch.set_default_tensor_type()`).
The :meth:`BasePrimary.to()` method modifies the instrument so that
subsequent simulations will be performed with the desired dtype.

.. code:: python

    >>> from pfhedge.instruments import BrownianStock
    >>>
    >>> stock = BrownianStock()
    >>> stock.to(torch.float64)
    BrownianStock(..., dtype=torch.float64)
    >>> stock.simulate(n_paths=2, time_horizon=2 / 250)
    >>> stock.spot
    tensor([[..., ..., ...],
            [..., ..., ...]], dtype=torch.float64)

One can also call ``to()`` to cast all the buffers that are already simulated
to the desired dtype.

.. code:: python

    >>> stock.to(float16).spot
    tensor([[..., ..., ...],
            [..., ..., ...]], dtype=torch.float16)

Derivative
^^^^^^^^^^

If one calls :meth:`BaseDerivative.to()`, its underlier gets the desired dtype.
As a result, the ``payoff`` also has the same dtype.

.. code:: python

    >>> from pfhedge.instruments import EuropeanOption
    >>>
    >>> _ = torch.manual_seed(42)
    >>> derivative = EuropeanOption(BrownianStock(), maturity=2 / 250)
    >>> derivative.to(torch.float64)
    EuropeanOption(
        ...
        (underlier): BrownianStock(..., dtype=torch.float64)
    )
    >>> derivative.simulate(n_paths=2)
    >>> derivative.ul().spot
    tensor([[..., ..., ...],
            [..., ..., ...]], dtype=torch.float64)
    >>> derivative.payoff()
    tensor([..., ...], dtype=torch.float64)

Instrument.device
-----------------

Primary
^^^^^^^

The simulation will be performed with a :class:`torch.device`
specified by :attr:`BasePrimary.device`.
The default device is the current device for the default tensor type
(see :func:`torch.set_default_tensor_type()`).
The :meth:`BasePrimary.to()` method modifies the instument so that
subsequent simulations will be performed on the desired device.

.. code:: python

    >>> from pfhedge.instruments import BrownianStock
    >>>
    >>> _ = torch.manual_seed(42)
    >>> stock = BrownianStock()
    >>> stock.to("cuda:0")
    BrownianStock(..., device='cuda:0')
    >>> stock.simulate(n_paths=2, time_horizon=2 / 250)
    >>> stock.spot
    tensor([[..., ..., ...],
            [..., ..., ...]], device='cuda:0')

One can also call ``to()`` to move all the buffers that are already simulated
to the desired device.

Derivative
^^^^^^^^^^

If one calls :meth:`BaseDerivative.to()`, its underlier gets the desired device.
As a result, the ``payoff`` is also on the same device.

.. code:: python

    >>> from pfhedge.instruments import EuropeanOption
    >>>
    >>> _ = torch.manual_seed(42)
    >>> derivative = EuropeanOption(BrownianStock(), maturity=2 / 250)
    >>> derivative.to("cuda:0")
    EuropeanOption(
        ...
        (underlier): BrownianStock(..., device='cuda:0')
    )
    >>> derivative.simulate(n_paths=2)
    >>> derivative.payoff()
    tensor([..., ...], device='cuda:0')

Now a :class:`pfhedge.nn.Hedger` module on a GPU can hedge the derivative
enjoying GPU-acceleration.

.. code:: python

    >>> from pfhedge.nn import Hedger
    >>>
    >>> derivative = EuropeanOption(...).to("cuda:0")
    >>> hedger = Hedger(...).to("cuda:0")
    >>> _ = hedger.fit(derivative)
    >>> hedger.price(derivative)
    torch.tensor(..., device='cuda:0')
