
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from optymus.search import line_search
from optymus.plots import plot_optim


class Optimizer:
    def __init__(self, f_obj, x0, grad=None, hess=None, method=None, tol=1e-5, max_iter=100):
        self.f_obj = f_obj
        self.grad = grad
        self.hess = hess
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

        if method is None:
            self.method = steepest_descent(f_obj, x0, grad, tol, max_iter)
        elif method == 'univariant':
            self.method = univariant(f_obj, x0, grad, tol, max_iter)
        elif method == 'powell':
            self.method = powell(f_obj, x0, grad, tol, max_iter)
        elif method == 'steepest_descent':
            self.method = steepest_descent(f_obj, x0, grad, tol, max_iter)
        elif method == 'fletcher_reeves':
            self.method = fletcher_reeves(f_obj, x0, grad, tol, max_iter)
        elif method == 'newton_raphson':
            self.method = newton_raphson(f_obj, x0, grad, hess, tol, max_iter)
        elif method == 'bfgs':
            self.method = bfgs(f_obj, x0, grad, tol, max_iter)
        else:
            raise Exception('Method not found. Please, choose one of the following methods: univariant, powell, steepest_descent, fletcher_reeves, newton_raphson, bfgs')
    
    def mimimize(self):
        return self.method
    
    def report(self):
        print(f"Method: {self.method['method_name']}")
        print(f"Minimum: {self.method['fmin']}")
        print(f"Number of iterations: {self.method['num_iter']}")
        print(f"Optimal point: {self.method['xopt']}")
    
    def plot(self):
        plot_optim(f = self.f_obj, x0=self.x0, method=self.method)


def univariant(f_obj, x0, grad, tol=1e-5, max_iter=100):
    x = x0.copy()
    n = len(x)                        
    u = np.identity(n)
    path = [x]
    alphas = []
    num_iter = 0       
    for _ in range(max_iter):             
        for i in range(n):
            v = u[i]
            r = line_search(f_obj, x, v)
            x = r['xopt']
            alphas.append(r['alpha'])
            path.append(x)
            if np.linalg.norm(grad(x)) < tol:
                result = {
                    'method_name': 'Univariant',
                    'xopt': x, 
                    'fmin': f_obj(x), 
                    'num_iter': num_iter, 
                    'path': np.array(path),
                    'alphas': np.array(alphas),
                    }
                return result
                
        num_iter += 1

def powell(f_obj, x0, grad, tol=1e-5, max_iter=100):
    x = x0.copy()
    def basis(i, n):
        return np.eye(n)[:, i-1]
    n = len(x)                   
    u = [basis(i,n) for i in range(1, n+1)]
    path = [x]
    alphas = []
    num_iter = 0       
    while np.linalg.norm(grad(x0)) > tol:            
        x_prime = x           
        for i in range(n):
            d = u[i]
            r = line_search(f_obj, x_prime, d)
            x_prime = r['xopt']
            alphas.append(r['alpha'])
            path.append(x_prime)
        for i in range(n-1):
            u[i] = u[i+1]
        u[n-1] = x_prime - x
        d = u[n-1]
        r = line_search(f_obj, x, d)
        x_prime = r['xopt']
        x0 = x_prime
        alphas.append(r['alpha'])
        path.append(x)
        fLast = f_obj(x)
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

def steepest_descent(f_obj, x0, grad, tol=1e-4, max_iter=100):
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


def fletcher_reeves(f_obj, x0, grad, tol=1e-5, maxiter=100):
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

def newton_raphson(f_obj, x0, grad, hess, tol=1e-5, maxiter=100):
    x = x0.copy()
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        g = grad(x)
        H = hess(x)  # Compute the hess at the current point x
        if np.linalg.norm(g) < tol:
            break
        p = np.linalg.solve(H, -g)
        ls = line_search(f_obj, x, p)
        alpha = ls['alpha']
        x = ls['xopt']
        alphas.append(alpha)
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'Newton-Raphson',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'path': np.array(path),
        'alphas': np.array(alphas)
    }
    return result