import sys
sys.path.append('./')

import numpy as np
from qsopt.search import bracket_minimum, golden_section, line_search
from qsopt.optim import method_optim
from qsopt.plots import plot_optim

def f(x):
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

def grad(x):
    return np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*x[0] + 2*x[1]**2 - 14, 2*x[0]**2 + 4*x[1]*(x[0] + x[1]**2 - 7) + 2*x[1] - 22])

def hess(x):
    return np.array([[12*x[0]**2 + 4*x[1] - 42, 4*x[0] + 4*x[1]], [4*x[0] + 4*x[1], 4*x[0] + 12*x[1]**2 - 26]])

x0 = np.array([0, 5])
d = np.array([3, 1.5])

method = bracket_minimum(f, x0)
#method = golden_section(f, x0)
#method = method_optim(method_name='powell', objective_function=f, initial_point=x0)
method = line_search(f, x0, d, step_size=0.01)
print(method)
#plot_optim(f, x0, method, path = False)


"""

def f(x): 
    return x**2-3*x+2

x0 = np.array(-5)
d = np.array(-1)

method = constant_step(f, x0)
#method = golden_search(f, x0)

import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
plt.plot(x, f(x))
plt.plot(x0, f(x0), 'ro')
plt.plot(method['xopt'], f(method['xopt']), 'bo')
plt.show()
"""