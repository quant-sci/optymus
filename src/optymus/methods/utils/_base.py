import jax.numpy as jnp

class BaseOptimizer:
    def __init__(self, f_obj=None, f_cons=None, args=(), args_cons=(), x0=None, tol=1e-4, learning_rate=0.01,
                 max_iter=100, verbose=True, maximize=False, gradient_type='fletcher_reeves', h_type='hessian', 
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.f_obj = f_obj
        self.f_cons = f_cons
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
        if self.maximize:
            return -self.f_obj(x, *self.args) + penalty
        return self.f_obj(x, *self.args) + penalty