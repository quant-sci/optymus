import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.search import line_search


class Univariate(BaseOptimizer):
    r"""Univariate Search Method

    In the Univariate Method, the search direction at iteration :math:`k` is defined by:

    .. math::
        \mathbf{d}_k = \mathbf{e}_k, \quad k = 1, \ldots, n

    where :math:`\mathbf{e}_k` is a vector with zero elements except at position :math:`k`, where the element is 1.

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
        Lerning rate for line search
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
        Lerning rate for line searchs
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)

        n = len(x)
        u = jnp.identity(n)
        path = [x]
        alphas = []
        num_iter = 0

        progres_bar = (
            tqdm(
                range(self.max_iter),
                desc=f"Univariant {num_iter}",
            )
            if self.verbose
            else range(self.max_iter)
        )

        for _ in progres_bar:
            if jnp.linalg.norm(jax.grad(self.penalized_obj)(x)) < self.tol:
                break
            for i in range(n):
                v = u[i]
                r = line_search(f=self.penalized_obj, x=x, d=v, learning_rate=self.learning_rate)
                x = r["xopt"]
                alphas.append(r["alpha"])
                path.append(x)
            num_iter += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "method_name": "Univariant" if not self.f_cons else "Univariant with Penalty",
            "xopt": x,
            "fmin": self.f_obj(x),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "alphas": jnp.array(alphas),
            "time": elapsed_time,
        }


def univariate(**kwargs):
    optimizer = Univariate(**kwargs)
    return optimizer.optimize()
