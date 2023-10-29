import sys
sys.path.append('./')

import time
import numpy as np
import sympy as sy
from sympy import *

from qsopt.plots import plot_optim

def golden_search(f, lbound, ubound, max_iter=100, tol=1e-5):
    phi = (np.sqrt(5)-1)/2  # Golden ratio
    num_iter = 0
    a, b = lbound, ubound
    beta = np.linalg.norm(b-a)
    alpha_e = a + (1 - phi)*beta
    alpha_d = a + (phi*beta)
    a_list = [a]
    b_list = [b]
    
    while beta > tol:
        #if num_iter > max_iter:
        #    break
        if f(alpha_e) < f(alpha_d):  # f(alpha) > f(alpha_d) to find the maximum
            b = alpha_d
        else:
            a = alpha_e

        beta = np.linalg.norm(b-a)
        alpha_e = a + (1 - phi)*beta
        alpha_d = a + (phi*beta)

        num_iter += 1
        a_list.append(a)
        b_list.append(b)
        
    alpha = (b + a) / 2
    fmin = f(alpha)

    result = {
        'alpha': alpha[np.argmin(alpha)],
        'fmin': fmin,
        'num_iter': num_iter,
        'path_a': np.array(a_list),
        'path_b': np.array(b_list),
    }
    return result

def line_search(objective_function, initial_point, d, search_method='golden_search'):
    def f(alpha): return objective_function(initial_point + alpha*d)
    if search_method == 'golden_search':
        a, b = np.array([-10, -10]), np.array([10, 10])
        r = golden_search(f, a, b)
    a = r['alpha']
    x_opt = initial_point + a*d
    result = {
        'alpha': a,
        'xopt': x_opt,
        'fmin': objective_function(x_opt),
        'initial_point': initial_point,
        'num_iter': r['num_iter'],
    }
    return result


def newton_raphson(objective_function, gradient, hessian, initial_point, tol=1e-5, maxiter=100):
    x = initial_point
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        g = gradient(x)
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
        'fopt': objective_function(x),
        'num_iter': num_iter,
        'path': np.array(path)
    }
    return result

def f(x):
    return np.sin(x[0]+x[1]) + (x[0]-x[1])**2-1.5*x[0]+2.5*x[1] 

def grad(x):
    return np.array([-2*x[0] + 2*x[1] + np.cos(x[0] + x[1]) + 2.5, 2*x[0] - 2*x[1] + np.cos(x[0] + x[1]) - 1.5])

def hess(x):
    return np.array([[-2*np.sin(x[0] + x[1]) - 2, -2*np.sin(x[0] + x[1]) + 2], [-2*np.sin(x[0] + x[1]) + 2, -2*np.sin(x[0] + x[1]) - 2]])

x0 = np.array([-2,3])
method = newton_raphson(f, grad, hess, x0)
plot_optim(f, x0, method)
#print(method['alphas'])