import jax
import jax.numpy as jnp
from optymus.search import line_search


def gradient_descent(f_obj, x0, tol=1e-4, max_iter=100):
    """
    Gradient Descent
    """
    x = x0.copy()
    g = jax.grad(f_obj)(x)
    d = -g
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(max_iter):
        if jnp.linalg.norm(g) < tol:
            break
        r = line_search(f_obj, x, d)
        alphas.append(r['alpha'])
        x = r['xopt']
        g = jax.grad(f_obj)(x)
        d = -g
        num_iter += 1
        path.append(x)
    return {
                'method_name': 'Gradient Descent',
                'xopt': x,
                'fmin': f_obj(x),
                'num_iter': num_iter,
                'path': jnp.array(path),
                'alphas': jnp.array(alphas),
                }


def conjugate_gradients(f_obj, x0, grad, tol=1e-5, maxiter=100, gradient_type=None):
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


def bfgs(f_obj, x0, tol=1e-5, maxiter=100):
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

def l_bfgs():
    """
    L-BFGS
    """

