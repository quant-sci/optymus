import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.search import line_search


def newton_raphson(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, tol=1e-5, learning_rate=0.01, max_iter=100, verbose=True, maximize=False):
    r"""Newton-Raphson method

    The Newton-Raphson method is a second-order optimization algorithm that uses
    the Hessian matrix to compute the step direction.

    We can minimize the objective function :math:`f` by solving the following
    equation:

    .. math::
        \nabla^2 f(x) p = -\nabla f(x)

    where :math:`\nabla^2 f(x)` is the Hessian matrix of :math:`f` evaluated at
    point :math:`x`, :math:`\nabla f(x)` is the gradient of :math:`f` evaluated
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
    maxiter : int
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
            Step sizes
    """
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

    grad = jax.grad(penalized_obj)
    hess = jax.hessian(penalized_obj)

    path = [x]
    alphas = []
    num_iter = 0

    progres_bar = tqdm(range(max_iter), desc=f'Newton-Raphson {num_iter}',) if verbose else range(max_iter)

    for _ in progres_bar:
        g = grad(x)
        H = hess(x)

        if jnp.linalg.norm(g) < tol:
            break

        d = jax.scipy.linalg.solve(H, -g)

        r = line_search(f=penalized_obj, x=x, d=d, learning_rate=learning_rate)
        x = r['xopt']

        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    return {
        'method_name': 'Newton-Raphson' if not f_cons else 'Newton-Raphson with Penalty',
        'x0': x0,
        'xopt': x,
        'fmin': f_obj(x, *args),
        'num_iter': num_iter,
        'path': jnp.array(path),
        'alphas': jnp.array(alphas),
        'time': elapsed_time
    }
