.. PFHedge documentation master file, created by
   sphinx-quickstart on Fri Jun 11 19:12:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pfnet-research/pfhedge

PFHedge Documentation
=====================

PFHedge is a `PyTorch <https://pytorch.org>`_-based framework for `Deep Hedging <https://arxiv.org/abs/1802.03042>`_.

|

Install:

.. code-block:: none

    pip install pfhedge

|

.. toctree::
   :maxdepth: 1
   :caption: API

   nn
   nn.functional
   instruments
   stochastic
   autogreek

|

.. toctree::
   :caption: Examples
   :hidden:
   :glob:

   examples/*
   Example Codes <https://github.com/pfnet-research/pfhedge/tree/main/examples>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/*

.. toctree::
   :caption: Development
   :hidden:

   GitHub <https://github.com/pfnet-research/pfhedge>
