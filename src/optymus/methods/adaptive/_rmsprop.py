import time

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.methods.utils._result import OptimizeResult

jax.config.update("jax_enable_x64", True)


class RMSProp(BaseOptimizer):
    r"""RMSprop optimizer

    RMSprop is an adaptive learning rate optimization algorithm that divides the learning rate
    by a running average of the squared gradients. It is particularly useful for non-stationary
    objectives.

    We can write the update rule for RMSprop as follows:

    .. math::
        E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2

        x_{t+1} = x_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t

    where :math:`E[g^2]_t` is the running average of the squared gradients, :math:`g_t` is the gradient,
    :math:`\beta` is the decay rate, :math:`\eta` is the learning rate, :math:`\epsilon` is a small constant
    to avoid division by zero, and :math:`t` is the current iteration.

    References
    ----------
    [1] Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4(2), 26-31.

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
    beta : float
        Decay rate
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
    eg2 : ndarray
        Running average of the squared gradients
    """

    def optimize(self):
        start_time = time.time()
        x = self.x0.astype(float)  # Ensure x0 is of a floating-point type

        grad = jax.grad(self.penalized_obj)
        Eg2 = jnp.zeros_like(x)
        path = [x]
        eg2_list = []
        f_history = [float(self.penalized_obj(x))]
        grad_norms = []
        num_iter = 0
        termination_reason = "max_iter_reached"

        progress_bar = tqdm(range(1, self.max_iter + 1), desc="RMSProp", disable=not self.verbose)

        for _ in progress_bar:
            g = grad(x)
            grad_norms.append(float(jnp.linalg.norm(g)))
            if jnp.linalg.norm(g) < self.tol:
                termination_reason = "gradient_norm_below_tol"
                break
            g = grad(x)
            if jnp.linalg.norm(g) < self.tol:
                termination_reason = "gradient_norm_below_tol"
                break
            Eg2 = self.beta1 * Eg2 + (1 - self.beta1) * g**2
            x = self.learning_rate * g / (jnp.sqrt(Eg2) + self.eps)
            x = self.project(x)

            path.append(x)
            eg2_list.append(Eg2)
            f_history.append(float(self.penalized_obj(x)))
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        return OptimizeResult({
            "method_name": "RMSprop" if not self.f_cons else "RMSprop with Penalty",
            "x0": self.x0,
            "xopt": x,
            "fmin": self.f_obj(x),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "eg2": jnp.array(eg2_list),
            "f_history": jnp.array(f_history),
            "grad_norms": jnp.array(grad_norms),
            "termination_reason": termination_reason,
            "time": elapsed_time,
        })


def rmsprop(**kwargs):
    optimizer = RMSProp(**kwargs)
    return optimizer.optimize()
