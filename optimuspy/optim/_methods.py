
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from qsopt.search import line_search
import numpy as np


def method_optim(method_name, objective_function, initial_point, gradient=None, hessian=None, tol=1e-5, max_iter=100):
    if method_name == 'univariant':
        return univariant(objective_function, gradient, initial_point, tol, max_iter)
    elif method_name == 'powell':
        return powell(objective_function, initial_point, gradient, tol, max_iter)
    elif method_name == 'steepest_descent':
        return steepest_descent(objective_function, gradient, initial_point, tol, max_iter)
    elif method_name == 'fletcher_reeves':
        return fletcher_reeves(objective_function, gradient, initial_point, tol, max_iter)
    elif method_name == 'bfgs':
        return bfgs(objective_function, gradient, initial_point, tol, max_iter)
    elif method_name == 'newton_raphson':
        return newton_raphson(objective_function, gradient, hessian, initial_point, tol, max_iter)
    else:
        raise ValueError('Unknown method: {}'.format(method_name))

def univariant(objective_function, gradient, initial_point, tol=1e-5, max_iter=100):
    x = initial_point.copy()
    n = len(x)                        
    u = np.identity(n)
    path = [x]
    alphas = []
    num_iter = 0       
    for _ in range(max_iter):             
        for i in range(n):
            v = u[i]
            r = line_search(objective_function, x, v)
            x = r['xopt']
            alphas.append(r['alpha'])
            path.append(x)
            if np.linalg.norm(gradient(x)) < tol:
                result = {
                    'method_name': 'Univariant',
                    'xopt': x, 
                    'fmin': objective_function(x), 
                    'num_iter': num_iter, 
                    'path': np.array(path),
                    'alphas': np.array(alphas),
                    }
                return result
                
        num_iter += 1

def powell(objective_function, x, gradient, tol=1e-5, max_iter=100):
    def basis(i, n):
        return np.eye(n)[:, i-1]
    n = len(x)                   
    u = [basis(i,n) for i in range(1, n+1)]
    path = [x]
    alphas = []
    num_iter = 0       
    while np.linalg.norm(gradient(x)) > tol:            
        x_prime = x           
        for i in range(n):
            d = u[i]
            r = line_search(objective_function, x_prime, d)
            x_prime = r['xopt']
            alphas.append(r['alpha'])
            path.append(x_prime)
        for i in range(n-1):
            u[i] = u[i+1]
        u[n-1] = x_prime - x
        d = u[n-1]
        r = line_search(objective_function, x, d)
        x_prime = r['xopt']
        x = x_prime
        alphas.append(r['alpha'])
        path.append(x)
        fLast = objective_function(x)
        num_iter += 1
    result = {
            'method_name': 'Powell',
            'xopt': x, 
            'fmin': fLast, 
            'num_iter': num_iter, 
            'path': np.array(path),
            'alphas': np.array(alphas),
            }
    return result

def steepest_descent(objective_function, gradient, initial_point, tol=1e-4, max_iter=100):
    x = initial_point.copy()
    g = gradient(x)
    d = -g
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        r = line_search(objective_function, x, d)
        alphas.append(r['alpha'])
        x = r['xopt']
        g = gradient(x)
        d = -g
        num_iter += 1
        path.append(x)
    result = {
                'method_name': 'Steepest Descent',
                'xopt': x, 
                'fmin': objective_function(x), 
                'num_iter': num_iter, 
                'path': np.array(path),
                'alphas': np.array(alphas),
                }        
    return result


def fletcher_reeves(objective_function, gradient, initial_point, tol=1e-5, maxiter=100):
    x = initial_point.copy()
    grad = gradient(x)
    direction = -grad
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        if np.linalg.norm(grad) <= tol:
            break
        r = line_search(objective_function, x, direction)
        x = r['xopt']
        new_grad = -gradient(x)
        if np.linalg.norm(new_grad) <= tol:
            break
        beta = np.dot(new_grad, new_grad) / np.dot(grad, grad)
        direction = -new_grad + beta * direction
        r = line_search(objective_function, x, direction)
        x = r['xopt']
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'Fletcher-Reeves',
        'xopt': x,
        'fmin': objective_function(x),
        'num_iter': num_iter,
        'path': np.array(path),
        'alphas': np.array(alphas)
        }
    return result


def bfgs(objective_function, grad, initial_point, tol=1e-5, maxiter=100):
    x = initial_point.copy()
    path = [x]
    alphas = []
    num_iter = 0
    for i in range(maxiter):
        Q, g = np.identity(len(x)), grad(x)
        x_new = line_search(objective_function, x, -np.dot(Q, g))['xopt']
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
        'fmin': objective_function(x),
        'num_iter': num_iter,
        'initial_point': initial_point,
        'path': np.array(path)
        }
    return result

def newton_raphson(objective_function, grad, hessian, initial_point, tol=1e-5, maxiter=100):
    x = initial_point.copy()
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        g = grad(x)
        H = hessian(x)  # Compute the Hessian at the current point x
        if np.linalg.norm(g) < tol:
            break
        p = np.linalg.solve(H, -g)
        ls = line_search(objective_function, x, p)
        alpha = ls['alpha']
        x = ls['xopt']
        alphas.append(alpha)
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'Newton-Raphson',
        'xopt': x,
        'fmin': objective_function(x),
        'num_iter': num_iter,
        'path': np.array(path),
        'alphas': np.array(alphas)
    }
    return result