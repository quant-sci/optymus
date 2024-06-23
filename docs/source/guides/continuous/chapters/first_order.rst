
First-Order Methods
===================

Steepest Descent (Gradient Descent)
-----------------------------------
In the Steepest Descent method (also know as Gradient Descent), the search direction at iteration \(k\) is defined by:

.. math::
    \mathbf{d}_k = -\nabla f(\mathbf{x}_k)

where :math:`\nabla f(\mathbf{x}_k)` is the gradient of the function at point :math:`\mathbf{x}_k`. The next point is calculated as:

.. math::
    \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k

where :math:`\alpha_k` is a scalar step size.

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='steepest_descent')

    # Print the solution
    results.print_report()

Conjugate Gradient Methods
--------------------------

Fletcher–Reeves Method
~~~~~~~~~~~~~~~~~~~~~~~

The Fletcher–Reeves method is defined as:

.. math::
    \mathbf{d}_k = -\nabla f(\mathbf{x}_k) + \beta_k \mathbf{d}_{k-1}

where :math:`\beta_k` is given by:

.. math::
    \beta_k = \frac{\|\nabla f(\mathbf{x}_k)\|^2}{\|\nabla f(\mathbf{x}_{k-1})\|^2}

The next point is calculated similarly as in the Gradient Descent method.

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='conjugate_gradient', gradient_type='fletcher_reeves')

    # Print the solution
    results.print_report()

Hestenes–Stiefel Method
~~~~~~~~~~~~~~~~~~~~~~~

In the Hestenes–Stiefel method, the parameter :math:`\beta_k` is calculated differently:

.. math::
    \beta_k = \frac{\nabla f(\mathbf{x}_k)^T (\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}_{k-1}))}{\mathbf{d}_{k-1}^T (\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}_{k-1}))}

The search direction and next point calculation follow the same pattern as above .

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='conjugate_gradient', gradient_type='hestenes_stiefel')

    # Print the solution
    results.print_report()

Polak–Ribiére Method
~~~~~~~~~~~~~~~~~~~~~

The Polak–Ribiére method calculates :math:`\beta_k` as follows:

.. math::
    \beta_k = \frac{\nabla f(\mathbf{x}_k)^T (\nabla f(\mathbf{x}_k) - \nabla f(\mathbf{x}_{k-1}))}{\|\nabla f(\mathbf{x}_{k-1})\|^2}

The search direction and next point are calculated in the same way as the other conjugate gradient methods.

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='conjugate_gradient', gradient_type='polak_ribiere')

    # Print the solution
    results.print_report()

Dai-Yuan Method
~~~~~~~~~~~~~~~

The Dai-Yuan method calculates :math:`\beta_k` as follows:

.. math::
    \beta_k = \frac{\nabla f(\mathbf{x}_k)^T \nabla f(\mathbf{x}_k)}{\nabla f(\mathbf{x}_{k-1})^T \nabla f(\mathbf{x}_{k-1})}

The search direction and next point are calculated in the same way as the other conjugate gradient methods.

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='conjugate_gradient', gradient_type='dai_yuan')

    # Print the solution
    results.print_report()

Quasi-Newton Methods
--------------------
Quasi-Newton methods seek to approximate the Hessian matrix to avoid the direct computation of second-order derivatives. These methods update the Hessian approximation :math:`B_k` at each iteration. In literature we can find this method classified as a second-order methods, but here we classify it as a first-order method because it does not require the computation of second-order derivatives, only and approximation of the Hessian matrix.

Broyden-Fletcher-Goldfarb-Shanno (BFGS) Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BFGS method updates the Hessian approximation as follows:

.. math::
    B_{k+1} = B_k + \frac{\mathbf{y}_k \mathbf{y}_k^T}{\mathbf{y}_k^T \mathbf{s}_k} - \frac{B_k \mathbf{s}_k \mathbf{s}_k^T B_k}{\mathbf{s}_k^T B_k \mathbf{s}_k}

where :math:`\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k` and :math:`\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)`.

The implementation with `optymus`:

.. code-block:: python

    from optymus import Optimizer

    # Define the objective function
    f = lambda x: x[0]**2 + x[1]**2

    # Define the initial point
    x0 = [0.0, 2.0]

    # Define the optimization problem
    results = Optimizer(f_obj=f, x0=x0, method='bfgs')

    # Print the solution
    results.print_report()

Limited-Memory BFGS (L-BFGS) Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The L-BFGS method is a memory-efficient version of the BFGS method. It stores only a few vectors of the most recent iterations to approximate the Hessian matrix.

The L-BFGS method approximates the Hessian matrix using the following formula:

.. math::
    B_{k+1} = (I - \rho_k s_k y_k^T) B_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T

where :math:`\rho_k = 1 / y_k^T s_k`.

.. note::
    The LBFGS is not implemented in `optymus` yet.
