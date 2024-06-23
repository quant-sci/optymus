
Second-Order Methods
====================

Newton-Raphson Method
---------------------

The Newton-Raphson method is defined by the update rule:

.. math::
    \mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)

where :math:`\nabla^2 f(\mathbf{x}_k)` is the Hessian matrix of second derivatives of :math:`f` at :math:`\mathbf{x}_k`. This method uses second-order information to find the minimum of the function.

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='newton_raphson')

    # Print the solution
    results.print_report()
