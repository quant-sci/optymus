import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

def adam(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=1000, verbose=True, maximize=False):
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
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

    grad = jax.grad(penalized_obj)
    m = jnp.zeros_like(x)  # First moment estimate
    v = jnp.zeros_like(x)  # Second moment estimate
    path = [x]
    lr = [learning_rate]
    num_iter = 0

    progress_bar = tqdm(range(1, max_iter+1), desc=f'Adam {num_iter}',) if verbose else range(1, max_iter+1)

    for t in progress_bar:
        if jnp.linalg.norm(grad(x)) < tol:
            break
        g = grad(x)  # Compute gradients
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        path.append(x)
        lr.append(learning_rate)
        num_iter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    return {
        'method_name': 'Adam' if not f_cons else 'Adam with Penalty',
        'x0':x0,
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': t,
        'path': jnp.array(path),
        'lr': jnp.array(lr),
        'time': elapsed_time
    }

def adagrad(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=100, verbose=True, maximize=False):
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
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

    grad = jax.grad(penalized_obj)

    g_sq_sum = jnp.zeros_like(x)
    path = [x]
    g_sum_list = []
    num_iter = 0

    progress_bar = tqdm(range(1, max_iter+1), desc=f'Adagrad {num_iter}',) if verbose else range(1, max_iter+1)

    for _ in progress_bar:
        if jnp.linalg.norm(grad(x)) < tol:
            break
        g = grad(x)
        g_sq_sum += g**2
        x -= learning_rate * g / (jnp.sqrt(g_sq_sum) + eps)

        path.append(x)
        g_sum_list.append(g_sq_sum)
        num_iter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    return {'method_name': 'Adagrad' if not f_cons else 'Adagrad with Penalty',
            'x0':x0,
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': _,
            'path': jnp.array(path),
            'g_sum': jnp.array(g_sum_list),
            'time': elapsed_time
            }


def rmsprop(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, beta=0.9, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=100, verbose=True, maximize=False):
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
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

    grad = jax.grad(penalized_obj)
    Eg2 = jnp.zeros_like(x)
    path = [x]
    eg2_list = []
    num_iter = 0

    progress_bar = tqdm(range(1, max_iter+1), desc=f'RMSProp {num_iter}',) if verbose else range(1, max_iter+1)

    for _ in progress_bar:
        if jnp.linalg.norm(grad(x)) < tol:
            break
        g = grad(x)
        if jnp.linalg.norm(g) < tol:
            break
        Eg2 = beta * Eg2 + (1 - beta) * g**2
        x = learning_rate * g / (jnp.sqrt(Eg2) + eps)

        path.append(x)
        eg2_list.append(Eg2)
        num_iter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    return {'method_name': 'RMSprop' if not f_cons else 'RMSprop with Penalty',
            'x0':x0,
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': _,
            'path': jnp.array(path),
            'eg2': jnp.array(eg2_list),
            'time': elapsed_time
            }

def adamax(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=100, verbose=True, maximize=False):
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
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

    grad = jax.grad(penalized_obj)
    m = jnp.zeros_like(x)
    u = jnp.zeros_like(x)
    path = [x]
    u_list = []
    num_iter = 0

    progress_bar = tqdm(range(1, max_iter+1), desc=f'Adamax {num_iter}',) if verbose else range(1, max_iter+1)

    for _ in progress_bar:
        if jnp.linalg.norm(grad(x)) < tol:
            break
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        u = jnp.maximum(beta2 * u, jnp.abs(g))
        x -= learning_rate * m / (u + eps)

        path.append(x)
        u_list.append(u)
        num_iter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    return {'method_name': 'Adamax' if not f_cons else 'Adamax with Penalty',
            'x0':x0,
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': num_iter,
            'path': jnp.array(path),
            'u': jnp.array(u_list),
            'time': elapsed_time
            }

def yogi(f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, beta1=0.9, beta2=0.999, eps=1e-3, tol=1e-5, learning_rate=0.01, max_iter=100, verbose=True, maximize=False):
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
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_cons is not None:
            for f_con in f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *args_cons)) ** 2)
        if maximize:
            return -f_obj(x, *args) + penalty
        return f_obj(x, *args) + penalty

    grad = jax.grad(penalized_obj)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    path = [x]
    v_list = []
    num_iter = 0

    progress_bar = tqdm(range(1, max_iter + 1), desc=f'Yogi {num_iter}',) if verbose else range(1, max_iter + 1)

    for t in progress_bar:
        if jnp.linalg.norm(grad(x)) < tol:
            break
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = v - (1 - beta2) * (g**2) * jnp.sign(v - g**2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        path.append(x)
        v_list.append(v)
        num_iter += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    return {'method_name': 'Yogi' if not f_cons else 'Yogi with Penalty',
            'x0':x0,
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': num_iter,
            'path': jnp.array(path),
            'v': jnp.array(v_list),
            'time': elapsed_time
            }
