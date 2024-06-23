
Second-Order Methods
====================

Newton-Raphson Method
---------------------

The Newton-Raphson method is defined by the update rule:

.. math::
    \mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)

where :math:`\nabla^2 f(\mathbf{x}_k)` is the Hessian matrix of second derivatives of :math:`f` at :math:`\mathbf{x}_k`. This method uses second-order information to find the minimum of the function .
