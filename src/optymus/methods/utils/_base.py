import jax.numpy as jnp


class BaseOptimizer:
    def __init__(
        self,
        f_obj=None,
        f_cons=None,
        bounds=(),
        args=(),
        args_cons=(),
        x0=None,
        tol=1e-6,
        learning_rate=0.01,
        max_iter=100,
        verbose=True,
        maximize=False,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def penalized_obj(self, x):
        penalty = 0.0

        if self.bounds:
            lower_bound, upper_bound = self.bounds
            x = jnp.clip(x, lower_bound, upper_bound)

        if self.f_cons is not None:
            for f_con in self.f_cons:
                penalty += jnp.sum(jnp.maximum(0, f_con(x, *self.args_cons)) ** 2)

        # Check if self.args is a tuple or list
        if isinstance(self.args, (tuple, list)):  # noqa
            obj_value = self.f_obj(x, *self.args)
        else:
            obj_value = self.f_obj(x, self.args)

        obj_value = jnp.squeeze(obj_value)

        if self.maximize:
            return -obj_value + penalty
        return obj_value + penalty
