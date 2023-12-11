import sys
sys.path.append('./')

import time
import numpy as np
import sympy as sy
from sympy import *

from qsopt.plots import plot_optim
from qsopt.search import line_search

def steepest_descent(objective_function, grad, initial_point, tol=1e-5, max_iter=100):
    x = initial_point.copy()
    num_iter = 0
    d = -grad(x)
    path = [x]
    alphas = []

    while np.linalg.norm(d) > tol:
        if max_iter < num_iter:
            break
        ls = line_search(objective_function, x, d)
        alphas.append(ls['alpha'])
        x = ls['xopt']
        d = -grad(x)

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

def f(x):
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

def grad(x):
    return np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*x[0] + 2*x[1]**2 - 14, 2*x[0]**2 + 4*x[1]*(x[0] + x[1]**2 - 7) + 2*x[1] - 22])

x0 = np.array([0, 5])
method = steepest_descent(f, grad, x0)
print(method['alphas'])
plot_optim(f, x0, method)