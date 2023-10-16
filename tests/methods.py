import numpy as np
import sys
sys.path.append('./')

from pyopt.optim import powell

def func(x):
    return x[0]**2 - 3*x[0]*x[1] + 4*x[1]**2 + x[0] - x[1]  

initial_point = np.array([1, 2])
direction = np.array([-1, -2])

pw = powell(func, initial_point)

print(pw)