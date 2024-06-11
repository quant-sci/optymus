import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.search import line_search


def univariant(f_obj=None, f_constr=None, x0=None, tol=1e-5, step_size=0.01, max_iter=100, verbose=True, maximize=False):
    """Univariant Search Method

    Parameters
    ----------
    f_obj : callable
        Objective function to minimize
    f_constr : callable
        Constraint function
    x0 : ndarray
        Initial guess
    tol : float
        Tolerance for stopping criteria
    step_size : float
        Step size
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
            Step sizes
    """
    x = x0.astype(float)
    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    n = len(x)
    u = jnp.identity(n)
    path = [x]
    alphas = []
    num_iter = 0

    progres_bar = tqdm(range(max_iter), desc=f'Univariant {num_iter}',) if verbose else range(max_iter)

    for _ in progres_bar:
        if jnp.linalg.norm(jax.grad(penalized_obj)(x)) < tol:
            break
        for i in range(n):
            v = u[i]
            r = line_search(f=penalized_obj, x=x, d=v, step_size=step_size)
            x = r['xopt']
            alphas.append(r['alpha'])
            path.append(x)
        num_iter += 1

    return {
            'method_name': 'Univariant',
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': num_iter,
            'path': jnp.array(path),
            'alphas': jnp.array(alphas),
            }

def powell(f_obj=None, f_constr=None, x0=None, tol=1e-5, step_size=0.01, max_iter=100, verbose=True, maximize=False):
    """Powell's Method

    Parameters
    ----------
    f_obj : callable
        Objective function to minimize
    f_constr : callable
        Constraint function
    x0 : ndarray
        Initial guess
    tol : float
        Tolerance for stopping criteria
    step_size : float
        Step size
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
            Step sizes
    """
    x = x0.astype(float)
    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    # Define gradient function using JAX's automatic differentiation
    grad = jax.grad(f_obj)

    # Function to create basis vectors
    def basis(i, n):
        return jnp.eye(n)[:, i-1]

    n = len(x)
    u = [basis(i, n) for i in range(1, n+1)]  # Initial basis vectors

    path = [x]
    alphas = []
    num_iter = 0

    progres_bar = tqdm(range(max_iter), desc=f'Gradient Descent {num_iter}',) if verbose else range(max_iter)

    for _ in progres_bar:
        # Perform line search along the basis vectors
        if jnp.linalg.norm(grad(x)) < tol:
            break

        x_prime = x.copy()
        for i in range(n):
            d = u[i]
            r = line_search(f=penalized_obj, x=x_prime, d=d, step_size=step_size)
            x_prime = r['xopt']
            alphas.append(r['alpha'])
            path.append(x_prime)

        # Update basis vectors
        for i in range(n-1):
            u[i] = u[i+1]
        u[n-1] = x_prime - x

        # Perform line search along the new direction
        d = u[n-1]
        r = line_search(f=penalized_obj, x=x, d=d, step_size=step_size)
        x_prime = r['xopt']

        x = x_prime
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1

    return {
        'method_name': 'Powell',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'path': jnp.array(path),
        'alphas': jnp.array(alphas),
    }
