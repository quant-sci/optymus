import numpy as np

def constant_step(func, initial_point, direction, delta_alpha=0.01, max_iter=1000, tol=1e-5):
    """
    Constant step method for finding the minimum of a function.
    
    INPUTS:
    function: function handle for objective function
    initial_point: initial guess for optimizer
    delta_alpha: step size (default: 0.01)
    max_iter: maximum number of iterations (default: 1000)
    tol: tolerance for convergence (default: 1e-5)
    
    OUTPUTS:
    xopt: optimal solution
    fopt: objective function value at xopt
    func_calls: number of function evaluations
    """
    x = initial_point
    f = func(x)
    num_iter = 0
    for k in range(max_iter):
        x_new = x + delta_alpha * direction
        f_new = func(x_new)
        
        if abs(f_new - f) < tol:
            break

        if f_new > f:
            x_new = x - delta_alpha * direction
            f_new = func(x_new)
        
        x = x_new
        f = f_new
        num_iter += 1
    
    xopt = x_new
    yopt = func(xopt)
    
    result = {
        'xopt': xopt,
        'yopt': yopt,
        'num_iter': num_iter,
        'initial_point': initial_point
    }

    return result

def bisection(func, initial_point, direction, delta_alpha=0.01, max_iter=1000, tol=1e-5):
    """
    Bisection method for finding the minimum of a function.
    
    INPUTS:
    f: function handle for objective function
    l_bound: lower bound of search interval
    u_bound: upper bound of search interval
    delta_alpha: step size for constant step method (default: 0.01)
    max_iter: maximum number of iterations (default: 1000)
    tol: tolerance for convergence (default: 1e-6)
    
    OUTPUTS:
    xopt: optimal solution
    fopt: objective function value at xopt
    func_calls: number of function evaluations
    """
    cs = constant_step(func, initial_point, direction=direction, delta_alpha=delta_alpha, max_iter=max_iter, tol=tol)
    l_bound = initial_point
    u_bound = cs['xopt']
    num_iter = 0
    eps = 1e-2
    
    while np.linalg.norm(np.array(u_bound) - np.array(l_bound)) > tol:
        beta = l_bound + u_bound
        mean_point = beta / 2        
        f_e = func(mean_point-eps)
        f_d = func(mean_point+eps)

        if f_e < f_d:
            l_bound = mean_point

        else:
            u_bound = mean_point
 
        num_iter += 1
    
    xopt = (l_bound + u_bound) / 2
    yopt = func(xopt)
    
    result = {
        'xopt': xopt,
        'yopt': yopt,
        'num_iter': num_iter,
        'initial_point': initial_point
    }

    return result

def golden_search(func, initial_point, l_bound, u_bound, max_iter=100, tol=1e-5):
    """
    Golden search method for finding the minimum of a function
    
    INPUTS:
    function: function handle for objective function
    l_bound: lower bound of search interval
    u_bound: upper bound of search interval
    delta_alpha: step size for constant step method (default: 0.01)
    max_iter: maximum number of iterations (default: 1000)
    tol: tolerance for convergence (default: 1e-6)
    
    OUTPUTS:
    xopt: optimal solution
    yopt: objective function value at xopt
    func_calls: number of function evaluations
    num_iter: number of iterations
    """
    phi = (1+np.sqrt(5)) / 2 # golden ratio
    num_iter = 0

    while np.linalg.norm(np.array(u_bound) - np.array(l_bound)) > tol or num_iter >= max_iter:
        beta = u_bound - l_bound
        alpha_e = u_bound - beta / phi
        alpha_d = l_bound + beta / phi
        f_e = func(alpha_e)
        f_d = func(alpha_d)

        if f_e < f_d:
            u_bound = alpha_d

        else:
            l_bound = alpha_e

        num_iter += 1
    
    xopt = (l_bound + u_bound) / 2
    yopt = func(xopt)
    
    result = {
        'xopt': xopt,
        'yopt': yopt,
        'num_iter': num_iter,
        'initial_point': initial_point
    }

    return result