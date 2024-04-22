import numpy as np

def mccormick_function():
    return lambda x: x[0]**[2]-3*x[0]*x[1]+4*x[1]**2+x[0]-x[1]

def rastrigin_function():
    return lambda x: 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))