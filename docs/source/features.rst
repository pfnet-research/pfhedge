.. _features:

pfhedge.features
================

`pfhedge.features` provides features for :class:`pfhedge.nn.Hedger`.

.. currentmodule:: pfhedge.features

Creation
--------

.. autosummary::
    :nosignatures:
    :toctree: generated

    get_feature
    list_feature_names

Derivatives Features
--------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    Moneyness
    LogMoneyness
    MaxMoneyness
    MaxLogMoneyness
    Barrier
    PrevHedge
    TimeToMaturity
    UnderlierSpot
    Variance
    Volatility

Tensor Creation Ops
-------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    Empty
    Ones
    Zeros

Containers
----------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: classtemplate.rst

    ModuleOutput
    FeatureList
