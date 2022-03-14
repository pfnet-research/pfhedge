.. role:: hidden
    :class: hidden-section

pfhedge.nn
==========

``pfhedge.nn`` provides :class:`torch.nn.Module` that are useful for Deep Hedging.

See `PyTorch Documentation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
for general usage of :class:`torch.nn.Module`.

.. currentmodule:: pfhedge

Hedger Module
-------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    nn.Hedger

Black-Scholes Layers
--------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    nn.BlackScholes
    nn.BSAmericanBinaryOption
    nn.BSEuropeanOption
    nn.BSEuropeanBinaryOption
    nn.BSLookbackOption

Whalley-Wilmott Layers
----------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    nn.WhalleyWilmott

Nonlinear Activations
---------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    nn.Clamp
    nn.LeakyClamp

Loss Functions
--------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    nn.HedgeLoss
    nn.EntropicLoss
    nn.EntropicRiskMeasure
    nn.ExpectedShortfall
    nn.IsoelasticLoss

Multi Layer Perceptron
----------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    nn.MultiLayerPerceptron

Other Modules
-------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    nn.Naked
    nn.SVIVariance
