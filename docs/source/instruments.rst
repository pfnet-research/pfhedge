.. role:: hidden
    :class: hidden-section

pfhedge.instruments
===================

.. currentmodule:: pfhedge

Base Class
----------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    instruments.BaseInstrument
    instruments.BasePrimary
    instruments.BaseDerivative
    instruments.OptionMixin
    instruments.BaseOption

Primary Instruments
-------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    instruments.BrownianStock
    instruments.HestonStock
    instruments.LocalVolatilityStock
    instruments.RoughBergomiStock
    instruments.CIRRate
    instruments.VasicekRate

Derivative Instruments
----------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    instruments.AmericanBinaryOption
    instruments.EuropeanBinaryOption
    instruments.EuropeanForwardStartOption
    instruments.EuropeanOption
    instruments.LookbackOption
    instruments.VarianceSwap
