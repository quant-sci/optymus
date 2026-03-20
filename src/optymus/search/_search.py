import jax.numpy as jnp
import numpy as np


def bracket_minimum(func, x0, dir, alpha=0.0, learning_rate=0.01, eps=1e-5, max_iter=1000):  # noqa
    alpha = jnp.array(alpha)
    learning_rate = jnp.array(learning_rate)
    eps = jnp.array(eps)
    best_alpha = alpha
    best_f = func(x0 + alpha * dir)
    for _ in range(max_iter):
        alpha = alpha + learning_rate
        f = func(x0 + alpha * dir)
        if f < best_f:
            best_f = f
            best_alpha = alpha
        f_prev = func(x0 + (alpha - eps) * dir)
        if f_prev < f:
            alpha_lower = alpha - learning_rate
            alpha_upper = alpha
            return alpha_lower, alpha_upper
    # max_iter exceeded — return best bracket found so far
    return best_alpha - learning_rate, best_alpha + learning_rate


def bracket(func, xa=0.0, xb=1.0, grow_limit=110.0, maxiter=1000):
    _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    _verysmall_num = 1e-21
    # convert to numpy floats if not already
    xa, xb = np.asarray([xa, xb])
    fa = func(*(xa,))
    fb = func(*(xb,))
    if fa < fb:  # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = func(*((xc,)))
    funcalls = 3
    iterations = 0
    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        denom = 2.0 * _verysmall_num if np.abs(val) < _verysmall_num else 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        msg = (
            "No valid bracket was found before the iteration limit was "
            "reached. Consider trying different initial points or "
            "increasing `maxiter`."
        )
        if iterations > maxiter:
            raise RuntimeError(msg)
        iterations += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(*((w,)))
            funcalls += 1
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif fw > fb:  # noqa
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func(*((w,)))
            funcalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = func(*((w,)))
            funcalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = func(*((w,)))
            funcalls += 1
            if fw < fc:
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(*((w,)))
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(*((w,)))
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    # three conditions for a valid bracket
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = xa < xb < xc or xc < xb < xa
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    msg = "The algorithm terminated without finding a valid bracket. " "Consider trying different initial points."
    if not (cond1 and cond2 and cond3):
        msg = "Bracketing values (xa, xb, xc) do not fulfill this " "requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
        raise RuntimeError(msg)

    return xa, xb, xc, fa, fb, fc, funcalls


def golden(func, brack=None, tol=1e-5, maxiter=1000):
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0], xb=brack[1])
    elif len(brack) == 3:
        xa, xb, xc = brack
        if xa > xc:  # swap so xa < xc can be assumed
            xc, xa = xa, xc
        if not ((xa < xb) and (xb < xc)):
            msg = "Bracketing values (xa, xb, xc) do not" " fulfill this requirement: (xa < xb) and (xb < xc)"
            raise ValueError(msg)
        fa = func(*((xa,)))
        fb = func(*((xb,)))
        fc = func(*((xc,)))
        if not ((fb < fa) and (fb < fc)):
            msg = (
                "Bracketing values (xa, xb, xc) do not fulfill" " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
            )
            raise ValueError(msg)
        funcalls = 3
    else:
        msg = "Bracketing interval must be length 2 or 3 sequence."
        raise ValueError(msg)

    _gR = 0.61803399  # golden ratio conjugate: 2.0/(1.0+sqrt(5.0))
    _gC = 1.0 - _gR
    x3 = xc
    x0 = xa
    if np.abs(xc - xb) > np.abs(xb - xa):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    f1 = func(*((x1,)))
    f2 = func(*((x2,)))
    funcalls += 2
    nit = 0

    for _ in range(maxiter):
        if np.abs(x3 - x0) <= tol * (np.abs(x1) + np.abs(x2)):
            break
        if f2 < f1:
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = func(*((x2,)))
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = func(*((x1,)))
        funcalls += 1

        nit += 1

    if f1 < f2:
        xmin = x1
        fval = f1
    else:
        xmin = x2
        fval = f2

    success = nit < maxiter and not (np.isnan(fval) or np.isnan(xmin))

    if success:
        message = (
            "\nOptimization terminated successfully;\n"
            "The returned value satisfies the termination criteria\n"
            f"(using tol = {tol} )"
        )
    else:
        message = "\nMaximum number of iterations exceeded" if nit >= maxiter else "NaN result encountered"

    return {"xmin": xmin, "fval": fval, "num_iter": nit, "function": func, "success": success, "message": message}


def line_search(f, x, d, learning_rate=0.01, tol=1e-5):
    a, b = bracket_minimum(func=f, x0=x, dir=d, learning_rate=learning_rate)
    result = golden(lambda alpha: f(x + alpha * d), brack=(a, b), tol=tol)
    alpha = result["xmin"]
    x_opt = x + alpha * d
    return {"method_name": "Line Search", "alpha": alpha, "xopt": x_opt, "fmin": f(x_opt), "num_iter": result["num_iter"], "function": f}


def backtracking_armijo(f, x, d, grad, alpha_init=1.0, c1=1e-4, rho=0.5, max_iter=50):
    """Backtracking line search with Armijo sufficient decrease condition.

    Finds alpha satisfying: f(x + alpha*d) <= f(x) + c1*alpha*(grad @ d)

    Parameters
    ----------
    f : callable
        Objective function
    x : ndarray
        Current point
    d : ndarray
        Search direction (must be a descent direction: grad @ d < 0)
    grad : ndarray
        Gradient at current point
    alpha_init : float
        Initial step size
    c1 : float
        Armijo condition parameter
    rho : float
        Backtracking factor
    max_iter : int
        Maximum number of backtracking steps

    Returns
    -------
    dict with alpha, xopt, fmin, num_iter
    """
    alpha = alpha_init
    f0 = f(x)
    slope = jnp.dot(grad, d)
    num_iter = 0

    for _ in range(max_iter):
        x_new = x + alpha * d
        if f(x_new) <= f0 + c1 * alpha * slope:
            break
        alpha = rho * alpha
        num_iter += 1

    x_opt = x + alpha * d
    return {"method_name": "Backtracking Armijo", "alpha": alpha, "xopt": x_opt, "fmin": f(x_opt), "num_iter": num_iter}


def wolfe_line_search(f, grad_f, x, d, grad, alpha_init=1.0, c1=1e-4, c2=0.9, max_iter=25):
    """Line search satisfying strong Wolfe conditions (Nocedal & Wright Algorithm 3.5/3.6).

    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable
        Gradient function
    x : ndarray
        Current point
    d : ndarray
        Search direction (must be a descent direction: grad @ d < 0)
    grad : ndarray
        Gradient at current point
    alpha_init : float
        Initial step size
    c1 : float
        Sufficient decrease parameter
    c2 : float
        Curvature condition parameter
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    dict with alpha, xopt, fmin, num_iter
    """
    f0 = f(x)
    slope0 = jnp.dot(grad, d)

    def _zoom(alpha_lo, alpha_hi, f_lo):
        """Zoom phase (Algorithm 3.6 from Nocedal & Wright)."""
        for _ in range(max_iter):
            # Bisection
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
            f_j = f(x + alpha_j * d)
            if f_j > f0 + c1 * alpha_j * slope0 or f_j >= f_lo:
                alpha_hi = alpha_j
            else:
                slope_j = jnp.dot(grad_f(x + alpha_j * d), d)
                if jnp.abs(slope_j) <= -c2 * slope0:
                    return alpha_j
                if slope_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j
                f_lo = f_j
        return 0.5 * (alpha_lo + alpha_hi)

    alpha_prev = 0.0
    alpha_curr = alpha_init
    f_prev = f0

    best_alpha = alpha_init
    best_f = float('inf')

    for i in range(1, max_iter + 1):
        f_curr = f(x + alpha_curr * d)

        if f_curr < best_f:
            best_f = f_curr
            best_alpha = alpha_curr

        if f_curr > f0 + c1 * alpha_curr * slope0 or (f_curr >= f_prev and i > 1):
            alpha = _zoom(alpha_prev, alpha_curr, f_prev)
            x_opt = x + alpha * d
            return {"method_name": "Wolfe Line Search", "alpha": alpha, "xopt": x_opt, "fmin": f(x_opt), "num_iter": i}

        slope_curr = jnp.dot(grad_f(x + alpha_curr * d), d)
        if jnp.abs(slope_curr) <= -c2 * slope0:
            x_opt = x + alpha_curr * d
            return {"method_name": "Wolfe Line Search", "alpha": alpha_curr, "xopt": x_opt, "fmin": f_curr, "num_iter": i}

        if slope_curr >= 0:
            alpha = _zoom(alpha_curr, alpha_prev, f_curr)
            x_opt = x + alpha * d
            return {"method_name": "Wolfe Line Search", "alpha": alpha, "xopt": x_opt, "fmin": f(x_opt), "num_iter": i}

        alpha_prev = alpha_curr
        f_prev = f_curr
        alpha_curr = 2.0 * alpha_curr  # Expand

    # Fallback to best alpha found
    x_opt = x + best_alpha * d
    return {"method_name": "Wolfe Line Search", "alpha": best_alpha, "xopt": x_opt, "fmin": f(x_opt), "num_iter": max_iter}
