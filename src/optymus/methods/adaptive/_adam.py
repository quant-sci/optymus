import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer

jax.config.update("jax_enable_x64", True)


class Adam(BaseOptimizer):
    r"""Adam optimization algorithm

    The Adam optimization algorithm is an extension of the stochastic gradient descent algorithm
    that computes adaptive learning rates for each parameter. It combines the advantages of two
    other extensions of stochastic gradient descent: AdaGrad and RMSProp.

    We can write the update rule for Adam as follows:

    .. math::
        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2

        \hat{m}_t = \frac{m_t}{1 - \beta_1^t}

        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

        x_{t+1} = x_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    where :math:`m_t` and :math:`v_t` are the first and second moment estimates, respectively,
    :math:`g_t` is the gradient, :math:`\beta_1` and :math:`\beta_2` are the exponential decay rates
    for the first and second moment estimates, respectively, :math:`\alpha` is the learning rate,
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

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)
        m = jnp.zeros_like(x)  # First moment estimate
        v = jnp.zeros_like(x)  # Second moment estimate
        path = [x]
        lr = [self.learning_rate]
        num_iter = 0

        progress_bar = (
            tqdm(
                range(1, self.max_iter + 1),
                desc=f"Adam {num_iter}",
            )
            if self.verbose
            else range(1, self.max_iter + 1)
        )

        for t in progress_bar:
            g = grad(x)
            if jnp.linalg.norm(g) < self.tol:
                break
            g = grad(x)  # Compute gradients
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g**2)
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            x = x - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.eps)

            path.append(x)
            lr.append(self.learning_rate)
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "method_name": "Adam" if not self.f_cons else "Adam with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x),
            "num_iter": t,
            "path": jnp.array(path),
            "lr": jnp.array(lr),
            "time": elapsed_time,
        }


def adam(**kwargs):
    optmizer = Adam(**kwargs)
    return optmizer.optimize()
