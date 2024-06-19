Optimization Methods
====================

This module contains the optimization methods for the optimization of the mathematical functions.

The methods are divided into four categories:

.. _zero_order

Zero Order Methods
------------------

.. autosummary::
    :toctree: .generated/

    optymus.methods.powell
    optymus.methods.univariant

.. _first_order

First Order Methods
-------------------

.. autosummary::
    :toctree: .generated/

    optymus.methods.bfgs
    optymus.methods.conjugate_gradient
    optymus.methods.gradient_descent

.. _second_order

Second Order Methods
--------------------

.. autosummary::
    :toctree: .generated/

    optymus.methods.newton_raphson

.. _adaptative

Adaptative Methods
------------------
    
.. autosummary::
    :toctree: .generated/

    optymus.methods.adagrad
    optymus.methods.adam
    optymus.methods.adamax
    optymus.methods.rmsprop
    optymus.methods.yogi