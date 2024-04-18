import numpy as np
from optymus.utils import line_search

def sgd(f_obj, x0, grad, tol=1e-4, max_iter=100):
    """
    Stochastic Gradient Descent
    """
    x = x0.copy()
    g = grad(x)
    d = -g
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        r = line_search(f_obj, x, d)
        alphas.append(r['alpha'])
        x = r['xopt']
        g = grad(x)
        d = -g
        num_iter += 1
        path.append(x)
    result = {
                'method_name': 'Steepest Descent',
                'xopt': x, 
                'fmin': f_obj(x), 
                'num_iter': num_iter, 
                'path': np.array(path),
                'alphas': np.array(alphas),
                }        
    return result


def conjugate_gradients(f_obj, x0, grad, tol=1e-5, maxiter=100, type='fletcher-reeves'):
    """
    Gradient Conjugates
    """
    x = x0.copy()
    grad = grad(x)
    direction = -grad
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        if np.linalg.norm(grad) <= tol:
            break
        r = line_search(f_obj, x, direction)
        x = r['xopt']
        new_grad = -grad(x)
        if np.linalg.norm(new_grad) <= tol:
            break
        beta = np.dot(new_grad, new_grad) / np.dot(grad, grad)
        direction = -new_grad + beta * direction
        r = line_search(f_obj, x, direction)
        x = r['xopt']
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'Fletcher-Reeves',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'path': np.array(path),
        'alphas': np.array(alphas)
        }
    return result


def bfgs(f_obj, x0, grad, tol=1e-5, maxiter=100):
    """
    BFGS
    """
    x = x0.copy()
    path = [x]
    alphas = []
    num_iter = 0
    for i in range(maxiter):
        Q, g = np.identity(len(x)), grad(x)
        x_new = line_search(f_obj, x, -np.dot(Q, g))['xopt']
        delta = x_new - x
        gamma = grad(x_new) - g
        if np.linalg.norm(delta) < tol:
            break
        if np.dot(delta, gamma) > 0:
            Q = np.dot(np.dot(delta, gamma), np.dot(delta, gamma)) / np.dot(delta, gamma)
        x = x_new
        path.append(x)
        num_iter += 1

    result = {
        'method_name': 'BFGS',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'x0': x0,
        'path': np.array(path)
        }
    return result

def l_bfgs():
    """
    L-BFGS
    """
    pass

