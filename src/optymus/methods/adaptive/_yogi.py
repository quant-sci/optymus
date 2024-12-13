import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer

jax.config.update("jax_enable_x64", True)


class Yogi(BaseOptimizer):
    r"""Yogi optimizer

    Yogi is an adaptive learning rate optimization algorithm that combines the advantages of
    the Adam and RMSprop optimization algorithms. It uses the sign of the gradient to adapt
    the learning rate.

    We can write the update rule for Yogi as follows:

    .. math::
        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

        v_t = v_{t-1} - (1 - \beta_2) (g_t^2) \text{sign}(v_{t-1} - g_t^2)

        \hat{m}_t = \frac{m_t}{1 - \beta_1^t}

        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

        x_{t+1} = x_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    where :math:`m_t` and :math:`v_t` are the first and second moment estimates, respectively,
    :math:`g_t` is the gradient, :math:`\beta_1` and :math:`\beta_2` are the exponential decay rates
    for the first and second moment estimates, respectively, :math:`\alpha` is the learning rate,
    :math:`\epsilon` is a small constant to avoid division by zero, and :math:`t` is the current iteration.

    References
    ----------
    [1] Zaheer, M., Reddi, S. J., Sachan, D. S., Kale, S., Kumar, S., & Hovy, E. (2018). Adaptive methods for nonconvex optimization. In Advances in Neural Information Processing Systems (pp. 8779-8788).

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
    v : ndarray
        Second moment estimates
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)
        m = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        path = [x]
        v_list = []
        num_iter = 0

        progress_bar = (
            tqdm(
                range(1, self.max_iter + 1),
                desc=f"Yogi {num_iter}",
            )
            if self.verbose
            else range(1, self.max_iter + 1)
        )

        for t in progress_bar:
            g = grad(x)
            if jnp.linalg.norm(g) < self.tol:
                break
            g = grad(x)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = v - (1 - self.beta2) * (g**2) * jnp.sign(v - g**2)
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            x = x - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.eps)

            path.append(x)
            v_list.append(v)
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "method_name": "Yogi" if not self.f_cons else "Yogi with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "v": jnp.array(v_list),
            "time": elapsed_time,
        }


def yogi(**kwargs):
    optimizer = Yogi(**kwargs)
    return optimizer.optimize()
