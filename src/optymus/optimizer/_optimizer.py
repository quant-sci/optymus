import jax  # noqa
from optymus.methods import (
    adagrad,
    adam,
    adamax,
    bfgs,
    conjugate_gradient,
    differential_evolution,
    newton_raphson,
    particle_swarm,
    powell,
    rmsprop,
    steepest_descent,
    univariate,
    yogi,
)
from optymus.optimizer.utils.report import Report
from optymus.plots import plot_optim

jax.config.update("jax_enable_x64", True)

METHODS = {
    "univariate": univariate,
    "powell": powell,
    "steepest_descent": steepest_descent,
    "conjugate_gradient": conjugate_gradient,
    "bfgs": bfgs,
    "newton_raphson": newton_raphson,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "adam": adam,
    "adamax": adamax,
    "yogi": yogi,
    "particle_swarm": particle_swarm,
    "differential_evolution": differential_evolution,
}


class Optimizer(Report):
    def __init__(self, f_obj=None, f_cons=None, x0=None, method="steepest_descent", **kwargs):
        """
        Initializes the Optimizer class.

        Args:
            f_obj (function): The objective function to be minimized.
            x0 (np.ndarray): The initial guess for the minimum.
            method (str, optional): The optimization method to use. Defaults to 'steepest_descent'.
        """
        self.f_obj = f_obj
        self.f_cons = f_cons
        self.x0 = x0
        self.method = method

        if self.method not in METHODS:
            msg = f"Method '{method}' not vailable. Available methods: {list(METHODS.keys())}"
            raise ValueError(msg)

        if self.f_obj is None:
            msg = "Objective function is required."
            raise ValueError(msg)

        # Run the optimization and store results
        self.opt = METHODS[self.method](f_obj=self.f_obj, f_cons=self.f_cons, x0=self.x0, **kwargs)

    def check_dimension(self):
        """Returns the dimension of the problem."""
        return len(self.x0)

    def get_results(self):
        """Returns the optimization results dictionary."""
        return self.opt

    def repr_info(self):
        return {
            "method_name": self.method,  # Ensure this is a string or valid method name
            "attributes": {
                "Initial Guess": self.x0,
                "Optimal Solution": self.opt.get("xopt", "N/A"),
                "Objective Function Value": self.opt.get("fmin", "N/A"),
                "Number of Iterations": self.opt.get("num_iter", "N/A"),
                "Time Elapsed": round(self.opt.get("time", "N/A"), 4),
            },
        }

    def report(self):
        """Generates a report of the optimization results."""
        return self.repr_html()

    def plot(self, **kwargs):
        """Plots the optimization path and function surface."""
        plot_optim(f_obj=self.f_obj, f_cons=self.f_cons, x0=self.x0, method=self.opt, **kwargs)
