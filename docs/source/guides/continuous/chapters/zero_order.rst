Zero-Order Methods
==================

Univariate Method
-----------------

In the Univariate Method, the search direction at iteration :math:`k` is defined by:

.. math::
    \mathbf{d}_k = \mathbf{e}_k, \quad k = 1, \ldots, n

where :math:`\mathbf{e}_k` is a vector with zero elements except at position :math:`k`, where the element is 1. This procedure is equivalent to modifying one variable at a time in the iterative process, meaning only the variable at position :math:`k` of the variable vector :math:`\mathbf{x}` is modified in iteration :math:`k`. For a problem with :math:`n` variables, if the position :math:`\mathbf{x}` has not converged to the solution :math:`\mathbf{x}^*` after :math:`n` iterations, a new cycle of iterations should start with the same directions used in the first cycle, and so on until convergence .

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='univariate')

    # Print the solution
    results.print_report()

Powell's Method
---------------

The Univariate Method is computationally inefficient and generally requires many iterations to reach a solution. One way to accelerate this process is to incorporate a new search direction, called the pattern movement, into the set of :math:`n` search directions at the end of each iterative cycle of :math:`n` iterations. During the first :math:`n` cycles, a pattern direction is incorporated at the end of each cycle into the set of :math:`n` search directions, replacing one of the discarded univariate directions. After :math:`n` cycles, no univariate direction should remain in the set of search directions. These new search directions were proposed by Powell and are obtained according to the expression below:

.. math::
    \mathbf{d}_j = \mathbf{x}_n - \mathbf{x}_0, \quad j = 1, \ldots, m

where :math:`\mathbf{x}_n` is the point obtained at the end of each cycle of :math:`n` iterations and :math:`\mathbf{x}_0` is the initial point. For each new cycle, if there is no convergence, a new pattern direction is created using the same procedure, i.e., the final point minus the initial point .

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='powell')

    # Print the solution
    results.print_report()