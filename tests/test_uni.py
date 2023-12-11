import sys
sys.path.append('./')
import time 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from qsopt.plots import plot_countour
from qsopt.optim import method_optim


import sympy as sy
from sympy import *
x1, x2 = sy.symbols("x1 x2")
func = x1**2-3*x1*x2+4*x2**2+x1-x2
print('Objective Function: ', func)
print('Gradient: ', Matrix([func]).jacobian(Matrix(list(func.free_symbols))))
print('Hessian: ', sy.hessian(func, varlist=[x1, x2]))


np.random.seed(1234)
f = lambda x: x[0]**[2]-3*x[0]*x[1]+4*x[1]**2+x[0]-x[1]
grad = lambda x: np.array([2*x[0]-3*x[1]+1, -3*x[0]+8*x[1]-1])
hess = lambda x: np.array([[2, -3], [-3, 8]])

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.view_init(40, 20)
ax.plot_surface(X, Y, Z, cmap='cividis', linewidth =0)

initial_point = [np.array([-2,2]), np.array([-1,-3])]

result = pd.DataFrame(columns=['Method', 'X0', 'XOPT', 'FMIN', 'NUM_ITER', 'TIME'])
for x0 in initial_point:
    METHODS_OPTIM = {
        'univariant': method_optim(method_name='univariant', objective_function=f, gradient=grad, initial_point=x0),
        'powell': method_optim(method_name='powell', objective_function=f, gradient=grad, initial_point=x0),
        'steepest_descent': method_optim(method_name='steepest_descent', objective_function=f, gradient=grad, initial_point=x0),
        'fletcher_reeves': method_optim(method_name='fletcher_reeves', objective_function=f, gradient=grad, initial_point=x0),
        'bfgs': method_optim(method_name='bfgs', objective_function=f, gradient=grad, initial_point=x0),
        'newton_raphson': method_optim(method_name='newton_raphson', objective_function=f, gradient=grad, hessian=hess, initial_point=x0),
        }
    methods = ['univariant', 'powell', 'steepest_descent', 'fletcher_reeves', 'bfgs', 'newton_raphson']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, m in enumerate(methods):
        start = time.time()
        method =  METHODS_OPTIM[m]
        end = time.time()
        time_elapsed = end - start
        r = pd.DataFrame([[method['method_name'], x0, method['xopt'], method['fmin'], method['num_iter'], time_elapsed]], 
                        columns=['Method', 'X0', 'XOPT', 'FMIN', 'NUM_ITER', 'TIME'])
        result = pd.concat([result, r], ignore_index=True, axis=0)
        plot_countour(f, x0, method, ax=axes[i], path = True)
    plt.tight_layout()
    plt.show()
result
