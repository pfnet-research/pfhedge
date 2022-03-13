.. role:: hidden
    :class: hidden-section

pfhedge.instruments
===================

.. currentmodule:: pfhedge

Base Class
----------

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    instruments.BaseInstrument
    instruments.BasePrimary
    instruments.BaseDerivative
    instruments.BaseOption

Primary Instruments
-------------------

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    instruments.BrownianStock
    instruments.HestonStock
    instruments.CIRRate

Derivative Instruments
----------------------

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    instruments.AmericanBinaryOption
    instruments.EuropeanBinaryOption
    instruments.EuropeanForwardStartOption
    instruments.EuropeanOption
    instruments.LookbackOption
    instruments.VarianceSwap
