.. _features:

Features
========

.. currentmodule:: pfhedge.features

Creation
--------

.. autosummary::
    :nosignatures:

    get_feature
    list_feature_names

Derivatives features
--------------------

.. autosummary::
    :nosignatures:
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

Tensor creation
---------------

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    Empty
    Ones
    Zeros

Containers
----------

.. autosummary::
    :nosignatures:
    :template: classtemplate.rst

    ModuleOutput
    FeatureList
