import streamlit as st
import time
import numpy as np
import sympy as sy
from sympy import *


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from qsopt.search import line_search, golden_search
from qsopt.optim import powell, steepest_descent
from qsopt.plots import plot_optim


def f(x): 
    return x[0]**[2]-3*x[0]*x[1]+4*x[1]**2+x[0]-x[1]

def grad(x):
    return np.array([-3*x[0]+8*x[1]-1, 2*x[0]-3*x[1]+1])



