import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.search import line_search


def gradient_descent(f_obj=None, f_constr=None, x0=None, tol=1e-4, max_iter=100, maximize=False):
    """
    Gradient Descent with constraints and maximization option.

    Args:
        f_obj (function): Objective function to be minimized or maximized.
        x0 (array): Initial guess for the variables.
        f_constr (function, optional): Constraint function to be satisfied.
        tol (float, optional): Tolerance for the gradient norm to determine convergence.
        max_iter (int, optional): Maximum number of iterations.
        maximize (bool, optional): If True, maximize the objective function.

    Returns:
        dict: Contains optimization results including optimal point, function value,
              number of iterations, path taken, and step sizes.
    """
    x = x0.astype(float)  # Ensure x0 is of a floating-point type
    sign = -1 if maximize else 1  # Multiply by -1 to maximize

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        return f_obj(x) + sign * penalty

    g = jax.grad(penalized_obj)(x)
    d = -sign * g
    path = [x]
    alphas = []
    num_iter = 0

    for _ in tqdm(range(max_iter), desc=f'Gradient Descent {num_iter}',):
        if jnp.linalg.norm(g) < tol:
            break
        r = line_search(f=penalized_obj, x=x, d=d, maximize=maximize)
        alphas.append(r['alpha'])
        x = r['xopt'].astype(float)  # Ensure xopt is of a floating-point type
        g = jax.grad(penalized_obj)(x)
        d = -sign * g
        num_iter += 1
        path.append(x)

    return {
        'method_name': 'Gradient Descent',
        'xopt': x,
        'fmin': f_obj(x) if not maximize else -f_obj(x),
        'num_iter': num_iter,
        'path': jnp.array(path),
        'alphas': jnp.array(alphas),
    }


def conjugate_gradients(f_obj=None, f_constr=None, x0=None, tol=1e-5, maxiter=100, gradient_type=None):
    """
    Gradient Conjugates
    """
    x = x0.copy()
    grad = jax.grad(f_obj)(x)
    direction = -grad
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        if jnp.linalg.norm(grad) <= tol:
            break
        r = line_search(f_obj, x, direction)
        x = r['xopt']
        new_grad = -grad(x)
        if jnp.linalg.norm(new_grad) <= tol:
            break
        if gradient_type is None:
            beta = jnp.dot(new_grad, new_grad) / jnp.dot(grad, grad)
        direction = -new_grad + beta * direction
        r = line_search(f_obj, x, direction)
        x = r['xopt']
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1
    return {
        'method_name': 'Conjugate Gradients',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'path': jnp.array(path),
        'alphas': jnp.array(alphas)
        }


def bfgs(f_obj=None, f_constr=None, x0=None, tol=1e-5, maxiter=100):
    """BFGS with JAX"""

    grad = jax.grad(f_obj)

    x = x0.copy()
    path = [x]
    num_iter = 0
    q = jnp.identity(len(x))  # Initial approximation of the inverse Hessian

    for _ in range(maxiter):
        g = grad(x)
        d = -jnp.dot(q, g)
        x_new = line_search(f_obj, x, d)['xopt']
        delta = x_new - x
        gamma = grad(x_new) - g

        if jnp.linalg.norm(delta) < tol:
            break

        rho = 1.0 / jnp.dot(delta, gamma)
        q = (jnp.eye(len(x)) - rho * jnp.outer(delta, gamma)) @ q
        q = q @ (jnp.eye(len(x)) - rho * jnp.outer(gamma, delta))
        q = q + rho * jnp.outer(delta, delta)  # BFGS update

        x = x_new
        path.append(x)
        num_iter += 1

    return {
        'method_name': 'BFGS',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'x0': x0,
        'path': jnp.array(path)
    }

def l_bfgs(f_obj=None, f_constr=None, x0=None):
    """
    L-BFGS
    """

