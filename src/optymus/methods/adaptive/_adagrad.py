import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer

jax.config.update("jax_enable_x64", True)


class AdaGrad(BaseOptimizer):
    r"""Adagrad optimizer

    Adagrad is an adaptive learning rate optimization algorithm that adapts the learning rate
    for each parameter based on the historical gradients. It is particularly useful for sparse
    data and non-stationary objectives.

    We can write the update rule for Adagrad as follows:

    .. math::
        g_{t} = \nabla f(x_t)

        G_{t} = G_{t-1} + g_{t}^2

        x_{t+1} = x_t - \frac{\eta}{\sqrt{G_{t} + \epsilon}} g_{t}

    where :math:`g_{t}` is the gradient, :math:`G_{t}` is the sum of the squares of the gradients,
    :math:`\eta` is the learning rate, :math:`\epsilon` is a small constant to avoid division by zero,
    and :math:`t` is the current iteration.

    References
    ----------
    [1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul), 2121-2159.

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
    g_sum : ndarray
        Sum of the squares of the gradients
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)

        g_sq_sum = jnp.zeros_like(x)
        path = [x]
        g_sum_list = []
        num_iter = 0

        progress_bar = (
            tqdm(
                range(1, self.max_iter + 1),
                desc=f"Adagrad {num_iter}",
            )
            if self.verbose
            else range(1, self.max_iter + 1)
        )

        for _ in progress_bar:
            g = grad(x)
            if jnp.linalg.norm(g) < self.tol:
                break
            g = grad(x)
            g_sq_sum += g**2
            x -= self.learning_rate * g / (jnp.sqrt(g_sq_sum) + self.eps)

            path.append(x)
            g_sum_list.append(g_sq_sum)
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "method_name": "Adagrad" if not self.f_cons else "Adagrad with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x),
            "num_iter": _,
            "path": jnp.array(path),
            "g_sum": jnp.array(g_sum_list),
            "time": elapsed_time,
        }


def adagrad(**kwargs):
    optimizer = AdaGrad(**kwargs)
    return optimizer.optimize()
