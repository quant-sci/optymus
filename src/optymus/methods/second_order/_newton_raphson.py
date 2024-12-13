import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.search import line_search


class NewtonRaphson(BaseOptimizer):
    r"""Newton-Raphson method with different matrix types

    The Newton-Raphson method is a second-order optimization algorithm that uses
    different matrices (Hessian, Fisher Information, Identity) to compute the step direction.

    We can minimize the objective function :math:`f` by solving the following
    equation:

    .. math::
        M(x) p = -\nabla f(x)

    where :math:`M(x)` is the chosen matrix (Hessian, Fisher Information, Identity) of :math:`f`
    evaluated at point :math:`x`, :math:`\nabla f(x)` is the gradient of :math:`f` evaluated
    at point :math:`x`, and :math:`p` is the step direction.

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
        Learning rate for line search
    max_iter : int
        Maximum number of iterations
    h_type : str
        Type of matrix to use ('hessian', 'fisher', 'identity')
    verbose : bool
        If True, prints progress
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

    def optimize(self, h_type="hessian"):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)
        hess = jax.hessian(self.penalized_obj)

        path = [x]
        alphas = []
        num_iter = 0

        progres_bar = (
            tqdm(
                range(self.max_iter),
                desc=f"Newton-Raphson {num_iter}",
            )
            if self.verbose
            else range(self.max_iter)
        )

        for _ in progres_bar:
            g = grad(x)

            if h_type == "hessian":
                M = hess(x)
            elif h_type == "fisher":
                M = -hess(x)
            elif h_type == "identity":
                M = jnp.eye(len(x))
            else:
                msg = f"Unknown h_type: {h_type}"
                raise ValueError(msg)

            if jnp.linalg.norm(g) < self.tol:
                break

            d = jnp.linalg.solve(M, -g)

            r = line_search(f=self.penalized_obj, x=x, d=d, learning_rate=self.learning_rate)
            x = r["xopt"]

            alphas.append(r["alpha"])
            path.append(x)
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "method_name": "Newton-Raphson" if not self.f_cons else "Newton-Raphson with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x, *self.args),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "alphas": jnp.array(alphas),
            "time": elapsed_time,
        }


def newton_raphson(htype='hessian', **kwargs):
    optimizer = NewtonRaphson(**kwargs)
    return optimizer.optimize(htype)
