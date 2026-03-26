import time

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.methods.utils._result import OptimizeResult


class Powell(BaseOptimizer):
    r"""Powell's Method

    Powell's method is a zero-order optimization method that uses a set of basis vectors to search for the minimum of a function.
    We can describe the method with the equation:

    .. math::
        x_{k+1} = x_k + \alpha_1 v_1 + \alpha_2 v_2 + \cdots + \alpha_n v_n

    where $x_k$ is the current point, $\alpha_i$ is the learning rate, and $v_i$ is the basis vector.

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
    dict
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

        grad = jax.grad(self.f_obj)

        # basis vectors
        def basis(i, n):
            return jnp.eye(n)[:, i - 1]

        n = len(x)
        u = [basis(i, n) for i in range(1, n + 1)]  # Initial basis vectors

        path = [x]
        alphas = []
        f_history = [float(self.penalized_obj(x))]
        grad_norms = []
        num_iter = 0
        termination_reason = "max_iter_reached"

        progress_bar = tqdm(range(self.max_iter), desc="Powell", disable=not self.verbose)

        for _ in progress_bar:
            # Perform line search along the basis vectors
            g_norm = float(jnp.linalg.norm(grad(x)))
            grad_norms.append(g_norm)
            if g_norm < self.tol:
                termination_reason = "gradient_norm_below_tol"
                break

            x_prime = x.copy()
            for i in range(n):
                d = u[i]
                r = self._do_line_search(x_prime, d)
                x_prime = self.project(r["xopt"])
                alphas.append(r["alpha"])
                path.append(x_prime)

            # Update basis vectors
            for i in range(n - 1):
                u[i] = u[i + 1]
            u[n - 1] = x_prime - x

            # Perform line search along the new direction
            d = u[n - 1]
            r = self._do_line_search(x, d)
            x_prime = self.project(r["xopt"])

            x = x_prime
            alphas.append(r["alpha"])
            path.append(x)
            f_history.append(float(self.penalized_obj(x)))
            num_iter += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        return OptimizeResult({
            "method_name": "Powell" if not self.f_cons else "Powell with Penalty",
            "xopt": x,
            "fmin": self.f_obj(x, *self.args),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "alphas": jnp.array(alphas),
            "f_history": jnp.array(f_history),
            "grad_norms": jnp.array(grad_norms),
            "termination_reason": termination_reason,
            "time": elapsed_time,
        })


def powell(**kwargs):
    optimizer = Powell(**kwargs)
    return optimizer.optimize()
