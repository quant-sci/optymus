import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

def adam(f_obj=None, f_constr=None, x0=None, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=1000, verbose=True, maximize=False):
    """Adam optimization algorithm"""
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

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
        'method_name': 'Adam' if not f_constr else 'Adam with Penalty',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': t,
        'path': jnp.array(path),
        'lr': jnp.array(lr),
        'time': elapsed_time
    }

def adagrad(f_obj=None, f_constr=None, x0=None, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=100, verbose=True, maximize=False):
    """Adagrad optimizer"""
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    grad = jax.grad(penalized_obj)

    g_sq_sum = jnp.zeros_like(x)
    path = [x]
    g_sum_list = []
    num_iter = 0

    progress_bar = tqdm(range(max_iter), desc=f'Adagrad {num_iter}',) if verbose else range(max_iter)

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
    return {'method_name': 'Adagrad' if not f_constr else 'Adagrad with Penalty',
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': _,
            'path': jnp.array(path),
            'g_sum': jnp.array(g_sum_list),
            'time': elapsed_time
            }


def rmsprop(f_obj=None, f_constr=None, x0=None, beta=0.9, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=100, verbose=True, maximize=False):
    """RMSprop optimizer"""
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    grad = jax.grad(penalized_obj)
    Eg2 = jnp.zeros_like(x)
    path = [x]
    eg2_list = []
    num_iter = 0

    progress_bar = tqdm(range(max_iter), desc=f'RMSProp {num_iter}',) if verbose else range(max_iter)

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
    return {'method_name': 'RMSprop',
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': _,
            'path': jnp.array(path),
            'eg2': jnp.array(eg2_list),
            'time': elapsed_time
            }

def adamax(f_obj=None, f_constr=None, x0=None, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-5, learning_rate=0.1, max_iter=100, verbose=True, maximize=False):
    """Adamax optimizer"""
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

    grad = jax.grad(penalized_obj)
    m = jnp.zeros_like(x)
    u = jnp.zeros_like(x)
    path = [x]
    u_list = []
    num_iter = 0

    progress_bar = tqdm(range(max_iter), desc=f'Adamax {num_iter}',) if verbose else range(max_iter)

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
    return {'method_name': 'Adamax' if not f_constr else 'Adamax with Penalty',
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': num_iter,
            'path': jnp.array(path),
            'u': jnp.array(u_list),
            'time': elapsed_time
            }

def yogi(f_obj=None, f_constr=None, x0=None, beta1=0.9, beta2=0.999, eps=1e-3, tol=1e-5, learning_rate=0.01, max_iter=100, verbose=True, maximize=False):
    """Yogi optimizer"""
    start_time = time.time()
    x = x0.astype(float)  # Ensure x0 is of a floating-point type

    def penalized_obj(x):
        penalty = 0.0
        if f_constr is not None:
            penalty = jnp.sum(jnp.maximum(0, f_constr(x)) ** 2)
        if maximize:
            return -f_obj(x) + penalty
        return f_obj(x) + penalty

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
    return {'method_name': 'Yogi' if not f_constr else 'Yogi with Penalty',
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': num_iter,
            'path': jnp.array(path),
            'v': jnp.array(v_list),
            'time': elapsed_time
            }
