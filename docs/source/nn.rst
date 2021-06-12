.. role:: hidden
    :class: hidden-section

pfhedge.nn
==========

These are the basic building blocks for graphs:

.. currentmodule:: pfhedge

Black-Scholes layers
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BlackScholes
    nn.BSAmericanBinaryOption
    nn.BSEuropeanOption
    nn.BSEuropeanBinaryOption
    nn.BSLookbackOption

Whalley-Wilmott layers
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.WhalleyWilmott

Nonlinear Activations
---------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Clamp
    nn.LeakyClamp

Loss Functions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.EntropicLoss
    nn.EntropicRiskMeasure
    nn.ExpectedShortfall
    nn.IsoelasticLoss

Multi Layer Perceptron
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.MultiLayerPerceptron

Other layers
------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Naked
