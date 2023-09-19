import sys
sys.path.append('./')

import warnings
warnings.filterwarnings('ignore')

import numpy as np

from utils.linear_search import constant_step, golden_section, bisection
from utils.plots import plot_optimization

def func(x):
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

initial_point = np.array([0, 5])
direction = np.array([3, 1.5])

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = func([X, Y])

constant_st = constant_step(function=func, initial_point=initial_point, direction=direction)
plot_optimization(X, Y, Z, constant_st)

bisec = bisection(function=func, initial_point=initial_point, direction=direction)
plot_optimization(X, Y, Z, bisec)

golden_sec = golden_section(function=func, initial_point=initial_point, direction=direction)
plot_optimization(X, Y, Z, golden_sec)