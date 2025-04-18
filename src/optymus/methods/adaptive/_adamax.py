import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer

jax.config.update("jax_enable_x64", True)


class Adamax(BaseOptimizer):
    r"""Adamax optimizer

    Adamax is an extension of the Adam optimization algorithm that uses the infinity norm
    of the gradients instead of the L2 norm. It is particularly useful for non-stationary
    objectives.

    We can write the update rule for Adamax as follows:

    .. math::
        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

        u_t = \max(\beta_2 u_{t-1}, |g_t|)

        x_{t+1} = x_t - \frac{\eta}{u_t + \epsilon} m_t

    where :math:`m_t` and :math:`u_t` are the first and infinity moment estimates, respectively,
    :math:`g_t` is the gradient, :math:`\beta_1` and :math:`\beta_2` are the exponential decay rates
    for the first and second moment estimates, respectively, :math:`\eta` is the learning rate,
    :math:`\epsilon` is a small constant to avoid division by zero, and :math:`t` is the current iteration.

    References
    ----------
    [1] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

    Parameters
    ----------
    f_obj : callable
        Objective function to minimize
    f_cons : list of callables
        List of constraint functions to minimize
    args : tuple
        Arguments to pass to the objective function
    args_cons : tuple
        Arguments to pass to the constraint functions
    x0 : ndarray
        Initial guess
    beta1 : float
        Exponential decay rate for the first moment estimates
    beta2 : float
        Exponential decay rate for the second moment estimates
    eps : float
        Small constant to avoid division by zero
    tol : float
        Tolerance for the norm of the gradient
    learning_rate : float
        Learning rate
    max_iter : int
        Maximum number of iterations
    verbose : bool
        Whether to display a progress bar
    maximize : bool
        Whether to maximize the objective function

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
    u : ndarray
        Infinity moment estimates
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)
        m = jnp.zeros_like(x)
        u = jnp.zeros_like(x)
        path = [x]
        u_list = []
        num_iter = 0

        progress_bar = (
            tqdm(
                range(1, self.max_iter + 1),
                desc=f"Adamax {num_iter}",
            )
            if self.verbose
            else range(1, self.max_iter + 1)
        )

        for _ in progress_bar:
            g = grad(x)
            if jnp.linalg.norm(g) < self.tol:
                break
            g = grad(x)
            m = self.beta1 * m + (1 - self.beta1) * g
            u = jnp.maximum(self.beta2 * u, jnp.abs(g))
            x -= self.learning_rate * m / (u + self.eps)

            path.append(x)
            u_list.append(u)
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "method_name": "Adamax" if not self.f_cons else "Adamax with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "u": jnp.array(u_list),
            "time": elapsed_time,
        }


def adamax(**kwargs):
    optmizer = Adamax(**kwargs)
    return optmizer.optimize()
