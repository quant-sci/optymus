import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.search import line_search


class BFGS(BaseOptimizer):
    r"""BFGS

    BFGS is a first-order optimization algorithm that uses the gradient of the
    objective function to compute the step direction.

    We can minimize the objective function :math:`f` by solving the following
    equation:

    .. math::
        x_{k+1} = x_k - \alpha_k d_k

    where :math:`x_k` is the current point, :math:`\alpha_k` is the step size,
    and :math:`d_k` is the search direction.

    The search direction :math:`d_k` is computed as follows:

    .. math::
        d_k = -B_k^{-1} \nabla f(x_k)

    where :math:`B_k` is an approximation of the inverse Hessian matrix.

    The inverse Hessian matrix :math:`B_k` is updated using the BFGS formula:

    .. math::
        B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{\delta_k \delta_k^T}{s_k^T \delta_k}

    where :math:`s_k = x_{k+1} - x_k`, :math:`\delta_k = \nabla f(x_{k+1}) - \nabla f(x_k)`.
    The step size :math:`\alpha_k` is computed using a line search algorithm.

    Parameters
    ----------
    f_obj : callable
        Objective function to minimize
    f_cons : callable
        Constraint function
    args : tuple
        Arguments for the objective function
    args_cons : tuple
        Arguments for the constraint function
    x0 : ndarray
        Initial guess
    tol : float
        Tolerance for stopping criteria
    learning_rate : float
        Step size
    max_iter : int
        Maximum number of iterations
    maximize : bool
        If True, maximize the objective function

    Returns
    -------
    method_name : str
        Method name
    xopt : ndarray
        Optimal point
    fmin : float
        Minimum value
    num_iter : int
        Number of iterations
    path : ndarray
        Path taken
    alphas : ndarray
        Step sizes
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        path = [x]
        alphas = []
        num_iter = 0
        q = jnp.identity(len(x))  # Initial approximation of the inverse Hessian

        progres_bar = (
            tqdm(
                range(self.max_iter),
                desc=f"BFGS {num_iter}",
            )
            if self.verbose
            else range(self.max_iter)
        )

        for _ in progres_bar:
            grad = jax.grad(self.penalized_obj)(x)
            d = jnp.dot(q, grad)
            r = line_search(f=self.penalized_obj, x=x, d=d, learning_rate=self.learning_rate)
            x_new = r["xopt"]
            delta = x_new - x
            gamma = jax.grad(self.penalized_obj)(x_new) - grad

            if jnp.linalg.norm(delta) < self.tol:
                break

            rho = 1.0 / jnp.dot(delta, gamma)
            q = (jnp.eye(len(x)) - rho * jnp.outer(delta, gamma)) @ q
            q = q @ (jnp.eye(len(x)) - rho * jnp.outer(gamma, delta))
            q = q + rho * jnp.outer(delta, delta)  # BFGS update

            x = x_new
            path.append(x)
            alphas.append(r["alpha"])
            num_iter += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "method_name": "BFGS" if not self.f_cons else "BFGS with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x, *self.args),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "alphas": jnp.array(alphas),
            "time": elapsed_time,
        }


def bfgs(**kwargs):
    optimizer = BFGS(**kwargs)
    return optimizer.optimize()
