import jax
import jax.numpy as jnp

from optymus.search import backtracking_armijo, line_search, wolfe_line_search


def _normalize_bounds(bounds, n_dims):
    """Normalize bounds to (lower_array, upper_array) jnp arrays.

    Accepts:
      - None / () / [] → (None, None)
      - [(lo, hi), ...] per-variable (scipy-style)
      - (lower_array, upper_array) legacy format
    """
    if bounds is None or (isinstance(bounds, (tuple, list)) and len(bounds) == 0):
        return None, None

    bounds_list = list(bounds)

    # Detect per-variable [(lo, hi), ...] format
    if len(bounds_list) == n_dims and all(
        isinstance(b, (tuple, list)) and len(b) == 2 for b in bounds_list
    ):
        lower = jnp.array([b[0] if b[0] is not None else -jnp.inf for b in bounds_list])
        upper = jnp.array([b[1] if b[1] is not None else jnp.inf for b in bounds_list])
    elif len(bounds_list) == 2:
        # (lower_array, upper_array) format
        lower = jnp.asarray(bounds_list[0], dtype=float)
        upper = jnp.asarray(bounds_list[1], dtype=float)
    else:
        msg = f"bounds must be [(lo, hi), ...] with len={n_dims} or (lower_array, upper_array), got len={len(bounds_list)}"
        raise ValueError(msg)

    if lower.shape != (n_dims,) or upper.shape != (n_dims,):
        msg = f"bounds arrays must have shape ({n_dims},), got {lower.shape} and {upper.shape}"
        raise ValueError(msg)

    if jnp.any(lower > upper):
        msg = "lower bounds must be <= upper bounds"
        raise ValueError(msg)

    return lower, upper


class BaseOptimizer:
    _default_line_search = "golden"

    def __init__(
        self,
        f_obj=None,
        f_cons=None,
        bounds=None,
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
        line_search_method=None,
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
        self.line_search_method = line_search_method if line_search_method is not None else self._default_line_search

        if self.bounds is not None and self.x0 is not None:
            self._lower_bounds, self._upper_bounds = _normalize_bounds(self.bounds, len(self.x0))
        else:
            self._lower_bounds, self._upper_bounds = None, None

    def project(self, x):
        if self._lower_bounds is None:
            return x
        return jnp.clip(x, self._lower_bounds, self._upper_bounds)

    def _do_line_search(self, x, d, grad=None):
        if self.line_search_method == "armijo":
            return backtracking_armijo(f=self.penalized_obj, x=x, d=d, grad=grad)
        elif self.line_search_method == "wolfe":
            return wolfe_line_search(
                f=self.penalized_obj, grad_f=jax.grad(self.penalized_obj),
                x=x, d=d, grad=grad,
            )
        else:
            return line_search(f=self.penalized_obj, x=x, d=d, learning_rate=self.learning_rate)

    def penalized_obj(self, x):
        penalty = 0.0

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
