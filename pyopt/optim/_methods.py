
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize

from pyopt.utils.tools import compute_gradient, line_search
from pyopt.search._search import golden_search


def univariant():
    pass

def powell(func, initial_point, tol=1e-5, maxiter=100):
    """
    Minimize a scalar function of one or more variables using Powell's method.

    Parameters:
    f : callable
        The objective function to be minimized.
    x0 : ndarray
        Initial guess. Array of real elements of size (n,), where n is the number of independent variables.
    tol : float, optional
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int, optional
        Maximum allowed number of iterations. Will default to n*1000, where n is the number of variables, if not set.

    Returns:
    res : OptimizeResult
        The optimization result represented as a dictionary. Important attributes are: x, success, message, fun, nit.
    """
    n = len(initial_point)
    j = 1
    direc = np.eye(n)
    x = initial_point.copy()
    fval = func(x)
    for i in range(maxiter):
        x_old = x
        fval_old = fval
        for k in range(n):
            yk = direc[:, k]
            alpha = minimize(lambda a: func(x + a * yk), x).x[0]
            x = x + alpha * yk
            fval = func(x)
        if np.abs(fval_old - fval) < tol:
            break
        dj = x - x_old
        alpha = minimize(lambda a: func(x - a * dj), x).x[0]
        x = x - alpha * dj
        fval = func(x)
        if np.abs(fval_old - fval) < tol:
            break
        direc[:, :-1] = direc[:, 1:]
        direc[:, -1] = dj
        j += 1
        if j > n:
            j = 1

    res = {'xopt': x,
           'yopt': fval, 
           'num_iter': i + 1,
           'initial_point': initial_point}
    return res  

def steepest_descent(func, initial_point, tol=1e-5, maxiter=100):
    """
    Minimize a scalar function using the steepest descent method.

    Parameters:
    func : callable
        The objective function to be minimized.
    initial_point : ndarray
        Initial guess. Array of real elements of size (n,), where n is the number of independent variables.
    tol : float, optional
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int, optional
        Maximum allowed number of iterations.

    Returns:
    res : dict
        The optimization result represented as a dictionary. Important attributes are: xopt, fopt, num_iter, initial_point.
    """
    n = len(initial_point)
    x = initial_point.copy()
    fval = func(x)
    for i in range(maxiter):
        gradient = np.zeros(n)
        for k in range(n):
            gradient[k] = minimize(lambda a: func(x - a * np.eye(n)[:, k]), 0).x[0]
        x = x - gradient
        fval = func(x)
        if np.linalg.norm(gradient) < tol:
            break

    res = {'xopt': x,
           'fopt': fval,
           'num_iter': i + 1,
           'initial_point': initial_point}
    return res

def fletcher_reeves(func, initial_point, tol=1e-5, maxiter=100):
    """
    Minimize a scalar function using the Fletcher-Reeves method.

    Parameters:
    func : callable
        The objective function to be minimized.
    initial_point : ndarray
        Initial guess. Array of real elements of size (n,), where n is the number of independent variables.
    tol : float, optional
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int, optional
        Maximum allowed number of iterations.

    Returns:
    res : dict
        The optimization result represented as a dictionary. Important attributes are: xopt, fopt, num_iter, initial_point.
    """
    n = len(initial_point)
    x = initial_point.copy()
    fval = func(x)
    gradient = np.zeros(n)
    direction = -gradient
    for i in range(maxiter):
        alpha = minimize(lambda a: func(x + a * direction), 0).x[0]
        x_new = x + alpha * direction
        gradient_new = np.array([minimize(lambda a: func(x_new + a * np.eye(n)[:, k]), 0).x[0] for k in range(n)])
        beta = np.dot(gradient_new, gradient_new) / np.dot(gradient, gradient)
        direction = -gradient_new + beta * direction
        x = x_new
        gradient = gradient_new
        fval = func(x)
        if np.linalg.norm(gradient) < tol:
            break

    res = {'xopt': x,
           'fopt': fval,
           'num_iter': i + 1,
           'initial_point': initial_point}
    return res

def bfgs(func, initial_point, tol=1e-5, maxiter=100):
    """
    Minimize a scalar function using the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm.

    Parameters:
    func : callable
        The objective function to be minimized.
    gradient : callable
        The gradient of the objective function.
    initial_point : ndarray
        Initial guess. Array of real elements of size (n,), where n is the number of independent variables.
    tol : float, optional
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int, optional
        Maximum allowed number of iterations.

    Returns:
    res : dict
        The optimization result represented as a dictionary. Important attributes are: xopt, fopt, num_iter, initial_point.
    """
    x = initial_point
    H = np.eye(len(x))  # Initialize the Hessian approximation as the identity matrix
    for i in range(maxiter):
        g = compute_gradient(func, x)
        if np.linalg.norm(g) < tol:
            break
        p = -np.dot(H, g)
        alpha = line_search(func, x, p)
        x_new = x + alpha * p
        s = x_new - x
        y = compute_gradient(func, x_new) - g

        # BFGS update
        rho = 1.0 / np.dot(y, s)
        A = np.eye(len(x)) - rho * np.outer(s, y)
        B = np.eye(len(x)) - rho * np.outer(y, s)
        H = np.dot(np.dot(A, H), B) + rho * np.outer(s, s)

        x = x_new

    result = {'xopt': x,
              'fopt': func(x),
              'num_iter': i + 1,
              'initial_point': initial_point}
    return result

def newton_raphson(func, gradient, hessian, initial_point, tol=1e-5, maxiter=100):
    """
    Minimize a scalar function using the Newton-Raphson algorithm.

    Parameters:
    func : callable
        The objective function to be minimized.
    gradient : callable
        The gradient of the objective function.
    hessian : callable
        The Hessian matrix of the objective function.
    initial_point : ndarray
        Initial guess. Array of real elements of size (n,), where n is the number of independent variables.
    tol : float, optional
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int, optional
        Maximum allowed number of iterations.

    Returns:
    res : dict
        The optimization result represented as a dictionary. Important attributes are: xopt, fopt, num_iter, initial_point.
    """
    x = initial_point
    for i in range(maxiter):
        g = gradient(x)
        H = hessian(x)
        if np.linalg.norm(g) < tol:
            break
        p = np.linalg.solve(H, -g)
        x_new = x + p
        x = x_new

    result = {'xopt': x,
              'fopt': func(x),
              'num_iter': i + 1,
              'initial_point': initial_point}
    return result