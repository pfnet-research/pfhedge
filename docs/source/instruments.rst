.. role:: hidden
    :class: hidden-section

pfhedge.instruments
===================

.. currentmodule:: pfhedge

Base Class
----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    instruments.BaseInstrument
    instruments.BasePrimary
    instruments.BaseDerivative
    instruments.BaseOption

Primary Instruments
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    instruments.BrownianStock
    instruments.HestonStock
    instruments.CIRRate

Derivative Instruments
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    instruments.EuropeanOption
    instruments.LookbackOption
    instruments.EuropeanBinaryOption
    instruments.AmericanBinaryOption
    instruments.VarianceSwap
