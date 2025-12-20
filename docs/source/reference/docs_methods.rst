Optimization Methods
====================

This module provides a comprehensive collection of optimization algorithms for solving
mathematical optimization problems. The methods are organized into seven categories based
on the type of information they use (derivative-free, gradient-based, Hessian-based) and
their algorithmic approach.

Zero-Order Methods
------------------

Derivative-free optimization methods that only require function evaluations.

.. autosummary::
    :toctree: .generated/

    optymus.methods.powell
    optymus.methods.univariate


First-Order Methods
-------------------

Gradient-based optimization methods that utilize first-order derivative information.

.. autosummary::
    :toctree: .generated/

    optymus.methods.bfgs
    optymus.methods.conjugate_gradient
    optymus.methods.steepest_descent


Second-Order Methods
--------------------

Methods that utilize second-order derivative (Hessian) information.

.. autosummary::
    :toctree: .generated/

    optymus.methods.newton_raphson


Adaptive Learning Rate Methods
------------------------------

Gradient-based methods with adaptive step-size mechanisms, commonly used in deep learning.

.. autosummary::
    :toctree: .generated/

    optymus.methods.adagrad
    optymus.methods.adam
    optymus.methods.adamax
    optymus.methods.rmsprop
    optymus.methods.yogi


Stochastic Methods
------------------

Probabilistic optimization algorithms that use randomness to explore the search space.

.. autosummary::
    :toctree: .generated/

    optymus.methods.cmaes
    optymus.methods.cross_entropy
    optymus.methods.simulated_annealing


Population-Based Methods
------------------------

Evolutionary and swarm intelligence algorithms that maintain a population of candidate solutions.

.. autosummary::
    :toctree: .generated/

    optymus.methods.particle_swarm
    optymus.methods.differential_evolution


Topological Optimization Methods
--------------------------------

Specialized methods for topology optimization problems.

.. autosummary::
    :toctree: .generated/

    optymus.methods.polymesher