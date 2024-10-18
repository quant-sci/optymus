import jax.numpy as jnp
from typing import Callable, Tuple, Optional
from dataclasses import dataclass

class BaseOptimizer:
    def __init__(self, f_obj=None, f_cons=None, bounds=Tuple[jnp.ndarray, jnp.ndarray], args=(), args_cons=(), x0=None, tol=1e-4, learning_rate=0.01,
                 max_iter=100, verbose=True, maximize=False, gradient_type='fletcher_reeves', h_type='hessian', 
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.f_obj = f_obj
        self.f_cons = f_cons
        self.bounds = bounds
        self.args = args
        self.args_cons = args_cons
        self.x0 = x0
        self.tol = tol
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.maximize = maximize
        self.gradient_type = gradient_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.h_type = h_type

    def penalized_obj(self, x):
        penalty = 0.0
        if self.f_cons is not None:
            for f_con in self.f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *self.args_cons)) ** 2)
        
        # Check if self.args is a tuple or list
        if isinstance(self.args, (tuple, list)):
            obj_value = self.f_obj(x, *self.args)
        else:
            obj_value = self.f_obj(x, self.args)
        
        if self.maximize:
            return -obj_value + penalty
        return obj_value + penalty
    

class BaseMMA(BaseOptimizer):
    # add extra arguments n and m to the __init__ method without repeating the arguments of the parent class
    def __init__(self, m_constraints, n_variables, n_iter, x_val, x_min, x_max, x_old_one,
                 x_old_two, f0_val, df0_dx, f_val, df_dx, lower_bound,upper_bound,
                 constant_a, coeff_a, coeff_c, coeff_d, move = 0.5, init_distance = 0.5,
                 decrease_distance = 0.7, increase_distance = 1.2, min_distance = 0.01, max_distance = 10,
                 appr_accuracy = 0.00001, bounds_alfa_beta = 0.1):
        super().__init__()
        self.n_variables = n_variables
        self.m_constraints = m_constraints
        self.n_iter = n_iter
        self.move = move
        self.init_distance = init_distance
        self.decrease_distance = decrease_distance
        self.increase_distance = increase_distance
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.appr_accuracy = appr_accuracy
        self.bounds_alfa_beta = bounds_alfa_beta
        self.x_val = x_val
        self.x_min = x_min
        self.x_max = x_max
        self.x_old_one = x_old_one
        self.x_old_two = x_old_two
        self.f0_val = f0_val
        self.df0_dx = df0_dx
        self.f_val = f_val
        self.df_dx = df_dx
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.constant_a = constant_a
        self.coeff_a = coeff_a
        self.coeff_c = coeff_c
        self.coeff_d = coeff_d