import jax  # noqa
from optymus.methods import (
    adagrad,
    adam,
    adamax,
    bfgs,
    conjugate_gradient,
    differential_evolution,
    lbfgs,
    newton_raphson,
    particle_swarm,
    powell,
    rmsprop,
    steepest_descent,
    univariate,
    yogi,
    cmaes,
    simulated_annealing,
    cross_entropy,
)
from optymus.optimizer.utils.constraints import (
    ConstraintMethodError,
    InfeasibleStartError,
    _as_list,
    run_barrier_method,
    run_penalty_method,
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
    "lbfgs": lbfgs,
    "newton_raphson": newton_raphson,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "adam": adam,
    "adamax": adamax,
    "yogi": yogi,
    "particle_swarm": particle_swarm,
    "differential_evolution": differential_evolution,
    "cmaes": cmaes,
    "simulated_annealing": simulated_annealing,
    "cross_entropy": cross_entropy,
}




class Optimizer(Report):
    def __init__(
        self,
        f_obj=None,
        f_cons=None,
        x0=None,
        method="steepest_descent",
        constraint_method=None,
        g_cons=None,
        h_cons=None,
        **kwargs,
    ):
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
        self.constraint_method = constraint_method
        self.g_cons = g_cons
        self.h_cons = h_cons

        if self.method not in METHODS:
            msg = f"Method '{method}' not vailable. Available methods: {list(METHODS.keys())}"
            raise ValueError(msg)

        if self.f_obj is None:
            msg = "Objective function is required."
            raise ValueError(msg)

        if self.constraint_method:
            constraint_method = self.constraint_method.lower()
            if constraint_method not in {"penalty", "barrier"}:
                msg = f"Constraint method '{self.constraint_method}' is not available. Use 'penalty' or 'barrier'."
                raise ConstraintMethodError(msg)

            g_cons = _as_list(self.g_cons if self.g_cons is not None else self.f_cons)
            h_cons = _as_list(self.h_cons)
            if not g_cons and not h_cons:
                msg = "Constraint method requires g_cons and/or h_cons."
                raise ConstraintMethodError(msg)

            inner_kwargs = kwargs.copy()
            penalty_r0 = inner_kwargs.pop("penalty_r0", 1.0)
            penalty_factor = inner_kwargs.pop("penalty_factor", 0.1)
            barrier_r0 = inner_kwargs.pop("barrier_r0", 1.0)
            barrier_factor = inner_kwargs.pop("barrier_factor", 0.1)
            max_outer_iter = inner_kwargs.pop("max_outer_iter", 10)
            constraint_tol = inner_kwargs.pop("constraint_tol", 1e-6)
            outer_tol = inner_kwargs.pop("outer_tol", 1e-6)
            barrier_type = inner_kwargs.pop("barrier_type", "log")
            barrier_eps = inner_kwargs.pop("barrier_eps", 1e-12)
            infeasible_penalty = inner_kwargs.pop("infeasible_penalty", 1e6)
            constraint_jit = inner_kwargs.pop("constraint_jit", False)
            warn_constraint_size = inner_kwargs.pop("warn_constraint_size", 10000)
            warn_slow_iter_s = inner_kwargs.pop("warn_slow_iter_s", None)
            warn_no_progress_iters = inner_kwargs.pop("warn_no_progress_iters", 3)
            warn_no_progress_tol = inner_kwargs.pop("warn_no_progress_tol", 1e-6)

            if constraint_method == "penalty":
                self.opt = run_penalty_method(
                    f_obj=self.f_obj,
                    g_cons=g_cons,
                    h_cons=h_cons,
                    x0=self.x0,
                    inner_method=METHODS[self.method],
                    constraint_jit=constraint_jit,
                    penalty_r0=penalty_r0,
                    penalty_factor=penalty_factor,
                    max_outer_iter=max_outer_iter,
                    constraint_tol=constraint_tol,
                    outer_tol=outer_tol,
                    warn_constraint_size=warn_constraint_size,
                    warn_slow_iter_s=warn_slow_iter_s,
                    warn_no_progress_iters=warn_no_progress_iters,
                    warn_no_progress_tol=warn_no_progress_tol,
                    inner_kwargs=inner_kwargs,
                )
            else:
                self.opt = run_barrier_method(
                    f_obj=self.f_obj,
                    g_cons=g_cons,
                    h_cons=h_cons,
                    x0=self.x0,
                    inner_method=METHODS[self.method],
                    constraint_jit=constraint_jit,
                    barrier_type=barrier_type,
                    barrier_r0=barrier_r0,
                    barrier_factor=barrier_factor,
                    penalty_r0=penalty_r0,
                    penalty_factor=penalty_factor,
                    max_outer_iter=max_outer_iter,
                    constraint_tol=constraint_tol,
                    outer_tol=outer_tol,
                    barrier_eps=barrier_eps,
                    infeasible_penalty=infeasible_penalty,
                    warn_constraint_size=warn_constraint_size,
                    warn_slow_iter_s=warn_slow_iter_s,
                    warn_no_progress_iters=warn_no_progress_iters,
                    warn_no_progress_tol=warn_no_progress_tol,
                    inner_kwargs=inner_kwargs,
                )
            self.f_cons = g_cons
        else:
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
