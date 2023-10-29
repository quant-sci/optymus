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


def fletcher_reeves(objective_function, grad, initial_point, tol=1e-5, maxiter=100):
    x = initial_point.copy()
    fval = objective_function(x)
    gradient = grad(x)
    path = [x]
    num_iter = 0
    for _ in range(maxiter):
        if np.linalg.norm(gradient) <= tol:
            break
        direction = -gradient
        r = line_search(objective_function, x, direction)
        x = r['xopt']
        new_gradient = grad(x)
        if np.linalg.norm(new_gradient) <= tol:
            break
        beta = np.dot(new_gradient, new_gradient) / np.dot(gradient, gradient)
        direction = -new_gradient + beta * direction
        r = line_search(objective_function, x, direction)
        x = r['xopt']
        path.append(x)
        num_iter += 1
    
    fval = objective_function(x)

    result = {
        'method_name': 'Fletcher-Reeves',
        'xopt': x,
        'fopt': fval,
        'num_iter': num_iter,
        'path': np.array(path)
        }
    return result

def f(x): 
    return x[0]**[2]-3*x[0]*x[1]+4*x[1]**2+x[0]-x[1]

def grad(x):
    return np.array([-3*x[0]+8*x[1]-1, 2*x[0]-3*x[1]+1])

x0 = np.array([4,2])
method = fletcher_reeves(f, grad, x0)
plot_optim(f, x0, method)
#print(method['alphas'])