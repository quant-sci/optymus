import time

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.methods.utils._result import OptimizeResult


class NewtonRaphson(BaseOptimizer):
    _default_line_search = "armijo"
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
        Type of matrix to use ('hessian', 'fisher', 'identity', 'bfgs').
        Use 'bfgs' for functions with custom_vjp (e.g., topology optimization).
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

        # Only compute exact Hessian if needed (avoids error with custom_vjp functions)
        hess = None
        if h_type in ("hessian", "fisher"):
            hess = jax.hessian(self.penalized_obj)

        # Initialize inverse Hessian approximation for BFGS
        B_inv = None
        if h_type == "bfgs":
            B_inv = jnp.eye(len(x))

        path = [x]
        alphas = []
        f_history = [float(self.penalized_obj(x))]
        grad_norms = []
        num_iter = 0
        termination_reason = "max_iter_reached"

        progres_bar = tqdm(range(self.max_iter), desc="Newton-Raphson", disable=not self.verbose)

        for _ in progres_bar:
            g = grad(x)
            grad_norms.append(float(jnp.linalg.norm(g)))

            if jnp.linalg.norm(g) < self.tol:
                termination_reason = "gradient_norm_below_tol"
                break

            # Compute search direction based on h_type
            if h_type == "hessian":
                M = hess(x)
                d = jnp.linalg.solve(M, -g)
            elif h_type == "fisher":
                M = -hess(x)
                d = jnp.linalg.solve(M, -g)
            elif h_type == "identity":
                d = -g  # No solve needed with identity matrix
            elif h_type == "bfgs":
                d = -jnp.dot(B_inv, g)  # No solve needed with inverse Hessian
            else:
                msg = f"Unknown h_type: {h_type}"
                raise ValueError(msg)

            r = self._do_line_search(x, d, g)
            x_new = self.project(r["xopt"])

            # BFGS inverse Hessian update
            if h_type == "bfgs":
                delta = x_new - x
                g_new = grad(x_new)
                gamma = g_new - g

                denom = jnp.dot(delta, gamma)
                if denom > 1e-10:  # Curvature condition check
                    rho = 1.0 / denom
                    I = jnp.eye(len(x))
                    B_inv = (I - rho * jnp.outer(delta, gamma)) @ B_inv
                    B_inv = B_inv @ (I - rho * jnp.outer(gamma, delta))
                    B_inv = B_inv + rho * jnp.outer(delta, delta)

            x = x_new

            alphas.append(r["alpha"])
            path.append(x)
            f_history.append(float(self.penalized_obj(x)))
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        method_suffix = f" ({h_type})" if h_type != "hessian" else ""
        penalty_suffix = " with Penalty" if self.f_cons else ""
        return OptimizeResult({
            "method_name": f"Newton-Raphson{method_suffix}{penalty_suffix}",
            "x0": self.x0,
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


def newton_raphson(htype='hessian', **kwargs):
    optimizer = NewtonRaphson(**kwargs)
    return optimizer.optimize(htype)
