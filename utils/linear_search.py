import numpy as np

def constant_step(function, initial_point, delta_alpha=0.01, max_iter=1000, tol=1e-5):
    """
    Constant step method for finding the minimum of a single-variable function.
    
    INPUTS:
    fun: function handle for objective function
    x0: initial guess for optimizer
    delta_alpha: step size (default: 0.01)
    maxit: maximum number of iterations (default: 1000)
    tol: tolerance for convergence (default: 1e-6)
    
    OUTPUTS:
    xopt: optimal solution
    fopt: objective function value at xopt
    func_calls: number of function evaluations
    """
    x = initial_point
    f = function(x)
    func_calls = 1
    for k in range(max_iter):
        x_new = x + delta_alpha
        f_new = function(x_new)
        func_calls += 1
        if abs(f_new - f) < tol:
            break
        x = x_new
        f = f_new
    xopt = x
    yopt = f
    return xopt, yopt, func_calls

def bisection(fun, a, b, delta_alpha=0.01, maxit=1000, tol=1e-6):
    """
    Bisection method for finding the minimum of a single-variable function.
    
    INPUTS:
    fun: function handle for objective function
    a: lower bound of search interval
    b: upper bound of search interval
    delta_alpha: step size for constant step method (default: 0.01)
    maxit: maximum number of iterations (default: 1000)
    tol: tolerance for convergence (default: 1e-6)
    
    OUTPUTS:
    xopt: optimal solution
    fopt: objective function value at xopt
    func_calls: number of function evaluations
    """
    fa = fun(a)
    fb = fun(b)
    func_calls = 2
    for k in range(maxit):
        x = (a + b) / 2
        fx = fun(x)
        func_calls += 1
        if abs(b - a) < tol:
            break
        if fx < fa:
            b = x
            fb = fx
        else:
            a = x
            fa = fx
    xopt = x
    fopt = fx
    _, _, func_calls_const = constant_step(fun, xopt, delta_alpha=delta_alpha, maxit=maxit, tol=tol)
    func_calls += func_calls_const
    return xopt, fopt, func_calls

def golden_section(function, l_bound, u_bound, delta_alpha=0.01, max_iter=1000, tol=1e-5):
    """
    Golden section method for finding the minimum of a single-variable function.
    
    INPUTS:
    function: function handle for objective function
    l_bound: lower bound of search interval
    u_bound: upper bound of search interval
    delta_alpha: step size for constant step method (default: 0.01)
    max_iter: maximum number of iterations (default: 1000)
    tol: tolerance for convergence (default: 1e-6)
    
    OUTPUTS:
    xopt: optimal solution
    """
    phi = (1+np.sqrt(5)) / 2 # golden ratio

    beta = u_bound - l_bound
    alpha_e = u_bound - beta / phi
    alpha_d = l_bound + beta / phi
    f_e = function(alpha_e)
    f_d = function(alpha_d)
    func_calls = 2

    for k in range(max_iter):
        if f_e < f_d:
            u_bound = alpha_d
            alpha_d = alpha_e
            beta = u_bound - l_bound
            alpha_e = u_bound - beta / phi
            f_d = f_e
            f_e = function(alpha_e)
        else:
            l_bound = alpha_e
            alpha_e = alpha_d
            beta = u_bound - l_bound
            alpha_d = l_bound + beta / phi
            f_e = f_d
            f_d = function(alpha_d)
        func_calls += 1
        if np.linalg.norm(np.array(u_bound) - np.array(l_bound)) < tol:
            break
    xopt = (l_bound + u_bound) / 2
    yopt = function(xopt)
    num_iter = k + 1
    _,_, func_calls_const = constant_step(function=function, initial_point=xopt, delta_alpha=delta_alpha, max_iter=max_iter, tol=tol)
    func_calls += func_calls_const
    return xopt, yopt, func_calls, num_iter