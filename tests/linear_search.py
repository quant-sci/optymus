import numpy as np
from scipy.optimize import minimize

import sys
sys.path.append('./')

from pyopt.search import constant_step
from pyopt.optim import powell
from pyopt.plots import plot_optimization


def func(x):
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

x0 = np.array([0, 5])
d = np.array([3, 1.5])

res = powell(func, x0)
print(res)