import jax.numpy as jnp


def bracket_minimum(func, x0, dir, alpha=0.0, learning_rate=0.01, eps=1e-5):
    alpha = jnp.array(alpha)
    learning_rate = jnp.array(learning_rate)
    eps = jnp.array(eps)
    while True:
        alpha = alpha + learning_rate
        f = func(x0 + alpha * dir)
        f_prev = func(x0 + (alpha - eps) * dir)
        if f_prev < f:
            alpha_lower = alpha - learning_rate
            alpha_upper = alpha
            return alpha_lower, alpha_upper

import scipy.optimize as opt


def line_search(f, x, d, learning_rate=0.01):
    a, b = bracket_minimum(func=f, x0=x, dir=d, learning_rate=learning_rate)
    alpha = opt.golden(lambda alpha: f(x + alpha * d), brack=(a, b), tol=1e-5)
    x_opt = x + alpha * d
    return {
        'method_name': 'Line Search',
        'alpha': alpha,
        'xopt': x_opt,
        'fmin': f(x_opt),
        'num_iter': 1,
        'function': f
    }
