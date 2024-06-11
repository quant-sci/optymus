import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.search import line_search


def gradient_descent(f_obj=None, f_constr=None, x0=None, tol=1e-4, step_size=0.01, max_iter=100, verbose=True, maximize=False):
    r"""Gradient Descent

    Gradient Descent is a first-order optimization algorithm that uses the
    gradient of the objective function to compute the step direction.

    We can minimize the objective function :math:`f` by solving the following
    equation:

    .. math::
        x_{k+1} = x_k - \alpha \nabla f(x_k)

    where :math:`x_k` is the current point, :math:`\alpha` is the step size,
    and :math:`\nabla f(x_k)` is the gradient of :math:`f` evaluated at point
    :math:`x_k`.

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
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    grad = jax.grad(penalized_obj)(x)
    d = grad
    path = [x]
    alphas = []
    num_iter = 0

    progres_bar = tqdm(range(max_iter), desc=f'Gradient Descent {num_iter}',) if verbose else range(max_iter)

    for _ in progres_bar:
        if jnp.linalg.norm(grad) < tol:
            break
        r = line_search(f=penalized_obj, x=x, d=d, step_size=step_size)
        alphas.append(r['alpha'])
        x = r['xopt'].astype(float)  # Ensure xopt is of a floating-point type
        grad = jax.grad(penalized_obj)(x)
        d = grad
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


def conjugate_gradient(f_obj=None, f_constr=None, x0=None, tol=1e-5, step_size=0.01, max_iter=100, verbose=True, gradient_type='fletcher_reeves', maximize=False):
    r"""Conjugate Gradient

    Conjugate Gradient is a first-order optimization algorithm that uses the
    gradient of the objective function to compute the step direction.

    We can minimize the objective function :math:`f` by solving the following
    equation:

    .. math::
        x_{k+1} = x_k - \alpha_k d_k

    where :math:`x_k` is the current point, :math:`\alpha_k` is the step size,
    and :math:`d_k` is the search direction.

    The search direction :math:`d_k` is computed as follows:

    .. math::
        d_k = -\nabla f(x_k) + \beta_k d_{k-1}

    where :math:`\nabla f(x_k)` is the gradient of :math:`f` evaluated at point
    :math:`x_k`, and :math:`\beta_k` is the conjugate gradient coefficient.

    We can compute beta using different formulas:

    - Fletcher-Reeves: :math:`\beta_k = \frac{\nabla x_k^{T} \nabla x_k}{\nabla x_{k-1}^{T} \nabla x_{k-1}}`
    - Polak-Ribiere: :math:`\beta_k = \frac{\nabla x_k^{T} (\nabla x_k - \nabla x_{k-1})}{\nabla x_{k-1}^T \nabla x_{k-1}}`
    - Hestnes-Stiefel: :math:`\beta_k = \frac{\nabla x_k^{T} (\nabla x_k - \nabla x_{k-1})}{\nabla s_{k-1}^{T}(\nabla x_{k} - \nabla x_{k-1})}` 
    - Dai-Yuan: :math:`\beta_k = \frac{\nabla x_{k}^{T} \nabla x_{k}}{\nabla s_{k-1}^{T}(\nabla x_{k} - \nabla x_{k-1})}`

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
    gradien_type: str
        'fletcher_reeves', 'polak_ribiere', 'hestnes_stiefel', 'dai_yuan'
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
    """  # noqa: E501
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    grad = jax.grad(penalized_obj)(x)
    d = grad
    path = [x]
    alphas = []
    num_iter = 0

    progres_bar = tqdm(range(max_iter), desc=f'Conjugate Gradient {num_iter}',) if verbose else range(max_iter)

    for _ in progres_bar:
        if jnp.linalg.norm(grad) <= tol:
            break
        r = line_search(f=penalized_obj, x=x, d=d, step_size=step_size)
        x = r['xopt']
        new_grad = jax.grad(penalized_obj)(x)
        if jnp.linalg.norm(new_grad) <= tol:
            break

        if gradient_type == 'fletcher_reeves':
            beta = jnp.dot(new_grad, new_grad) / jnp.dot(grad, grad)

        elif gradient_type == 'polak_ribiere':
            beta = jnp.dot(new_grad, new_grad - grad) / jnp.dot(grad, grad)

        elif gradient_type == 'hestnes_stiefel':
            beta = jnp.dot(new_grad, new_grad-grad) / jnp.dot(d, new_grad-grad)

        elif gradient_type == 'dai_yuan':
            beta = jnp.dot(new_grad, new_grad) / jnp.dot(d, new_grad-grad)

        d = new_grad + beta * d
        r = line_search(f=penalized_obj, x=x, d=d, step_size=step_size)
        x = r['xopt']
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1

    return {
        'method_name': f'Conjugate Gradients ({gradient_type})',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'path': jnp.array(path),
        'alphas': jnp.array(alphas),
        }


def bfgs(f_obj=None, f_constr=None, x0=None, tol=1e-5, step_size=0.01, max_iter=100, verbose=True, maximize=False):
    """BFGS with JAX"""
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    path = [x]
    num_iter = 0
    q = jnp.identity(len(x))  # Initial approximation of the inverse Hessian

    progres_bar = tqdm(range(max_iter), desc=f'BFGS {num_iter}',) if verbose else range(max_iter)

    for _ in progres_bar:
        grad = jax.grad(penalized_obj)(x)
        d = jnp.dot(q, grad)
        x_new = line_search(f=penalized_obj, x=x, d=d, step_size=step_size)['xopt']
        delta = x_new - x
        gamma = jax.grad(penalized_obj)(x_new) - grad

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

