import numpy as np

def compute_gradient(func, x, epsilon=1e-5):
    """
    Compute the gradient of a scalar function using finite differences.

    Parameters:
    func : callable
        The objective function to compute the gradient for.
    x : ndarray
        The point at which to compute the gradient.
    epsilon : float, optional
        A small value for the finite difference.

    Returns:
    gradient : ndarray
        The gradient vector at the given point x.
    """
    n = len(x)
    gradient = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        gradient[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
    
    return gradient

def line_search(func, x, p, alpha0=1.0, c=0.1, rho=0.9, maxiter=100):
    alpha = alpha0
    for i in range(maxiter):
        if func(x + alpha * p) <= func(x) + c * alpha * np.dot(compute_gradient(func, x), p):
            return alpha
        alpha *= rho
    return alpha

