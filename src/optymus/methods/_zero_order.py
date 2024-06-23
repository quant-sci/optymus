import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.search import line_search


def univariant(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, tol=1e-5, learning_rate=0.01, max_iter=100, verbose=True, maximize=False):
    r"""Univariant Search Method

    The univariant search method is a zero-order optimization method that uses a set of basis vectors to search for the minimum of a function.

    We can describe the method with the equation:

    .. math::

        x_{k+1} = x_k + \alpha_k v_k

    where $x_k$ is the current point, $\alpha_k$ is the learning rate, and $v_k$ is the basis vector.

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
    start_time = time.time()
    x = x0.astype(float)
    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

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
            r = line_search(f=penalized_obj, x=x, d=v, learning_rate=learning_rate)
            x = r['xopt']
            alphas.append(r['alpha'])
            path.append(x)
        num_iter += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    return {
            'method_name': 'Univariant' if not f_cons else 'Univariant with Penalty',
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': num_iter,
            'path': jnp.array(path),
            'alphas': jnp.array(alphas),
            'time': elapsed_time,
            }

def powell(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, tol=1e-5, learning_rate=0.01, max_iter=100, verbose=True, maximize=False):
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
    start_time = time.time()
    x = x0.astype(float)
    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

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

    progres_bar = tqdm(range(max_iter), desc=f'Powell {num_iter}',) if verbose else range(max_iter)

    for _ in progres_bar:
        # Perform line search along the basis vectors
        if jnp.linalg.norm(grad(x)) < tol:
            break

        x_prime = x.copy()
        for i in range(n):
            d = u[i]
            r = line_search(f=penalized_obj, x=x_prime, d=d, learning_rate=learning_rate)
            x_prime = r['xopt']
            alphas.append(r['alpha'])
            path.append(x_prime)

        # Update basis vectors
        for i in range(n-1):
            u[i] = u[i+1]
        u[n-1] = x_prime - x

        # Perform line search along the new direction
        d = u[n-1]
        r = line_search(f=penalized_obj, x=x, d=d, learning_rate=learning_rate)
        x_prime = r['xopt']

        x = x_prime
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    return {
        'method_name': 'Powell' if not f_cons else 'Powell with Penalty',
        'xopt': x,
        'fmin': f_obj(x, *args),
        'num_iter': num_iter,
        'path': jnp.array(path),
        'alphas': jnp.array(alphas),
        'time': elapsed_time,
    }
