import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.search import line_search


class ConjugateGradient(BaseOptimizer):
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
        Step size
    max_iter : int
        Maximum number of iterations
    gradien_type: str
        'fletcher_reeves', 'polak_ribiere', 'hestnes_stiefel', 'dai_yuan'
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

    def optimize(self, gradient_type):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)(x)
        d = grad
        path = [x]
        alphas = []
        num_iter = 0

        progres_bar = (
            tqdm(
                range(self.max_iter),
                desc=f"Conjugate Gradient {num_iter}",
            )
            if self.verbose
            else range(self.max_iter)
        )

        for _ in progres_bar:
            if jnp.linalg.norm(grad) <= self.tol:
                break
            r = line_search(f=self.penalized_obj, x=x, d=d, learning_rate=self.learning_rate)
            x = r["xopt"]
            new_grad = jax.grad(self.penalized_obj)(x)
            if jnp.linalg.norm(new_grad) <= self.tol:
                break

            if gradient_type == "fletcher_reeves":
                beta = jnp.dot(new_grad, new_grad) / jnp.dot(grad, grad)

            elif gradient_type == "polak_ribiere":
                beta = jnp.dot(new_grad, new_grad - grad) / jnp.dot(grad, grad)

            elif gradient_type == "hestnes_stiefel":
                beta = jnp.dot(new_grad, new_grad - grad) / jnp.dot(d, new_grad - grad)

            elif gradient_type == "dai_yuan":
                beta = jnp.dot(new_grad, new_grad) / jnp.dot(d, new_grad - grad)

            d = new_grad + beta * d
            r = line_search(f=self.penalized_obj, x=x, d=d, learning_rate=self.learning_rate)
            x = r["xopt"]
            alphas.append(r["alpha"])
            path.append(x)
            num_iter += 1
        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "method_name": f"Conjugate Gradients ({gradient_type})"
            if not self.f_cons
            else f"Conjugate Gradients ({gradient_type}) with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x, *self.args),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "alphas": jnp.array(alphas),
            "time": elapsed_time,
        }


def conjugate_gradient(gradient_type="fletcher_reeves", **kwargs):
    optimizer = ConjugateGradient(**kwargs)
    return optimizer.optimize(gradient_type)
