import pytest
import jax.numpy as jnp
from optymus.methods import (
    steepest_descent,
    conjugate_gradient,
    bfgs,
    lbfgs,
    newton_raphson,
    univariate,
    powell,
    adam,
    adamax,
    adagrad,
    yogi,
    cmaes,
    cross_entropy,
    simulated_annealing,
    differential_evolution,
    particle_swarm,
)


f_obj = lambda x: x[0] ** 2 + x[1] ** 2
x0 = jnp.array([0.0, 1.0])
common_kwargs = dict(f_obj=f_obj, x0=x0, tol=1e-5, learning_rate=0.1, max_iter=100, verbose=False)


# --- Gradient-based methods: f_history, grad_norms, termination_reason ---

@pytest.mark.parametrize("method_fn,extra_kwargs", [
    (steepest_descent, {}),
    (bfgs, {}),
    (lbfgs, {}),
    (newton_raphson, {}),
    (conjugate_gradient, {"gradient_type": "fletcher_reeves"}),
    (univariate, {}),
    (powell, {}),
])
def test_gradient_method_diagnostics(method_fn, extra_kwargs):
    result = method_fn(**common_kwargs, **extra_kwargs)

    assert "f_history" in result
    assert "grad_norms" in result
    assert "termination_reason" in result

    assert len(result["f_history"]) >= 1
    assert len(result["grad_norms"]) >= 1
    assert result["termination_reason"] in ("gradient_norm_below_tol", "step_size_below_tol", "max_iter_reached")


@pytest.mark.parametrize("method_fn,extra_kwargs", [
    (steepest_descent, {}),
    (bfgs, {}),
    (lbfgs, {}),
    (newton_raphson, {}),
    (conjugate_gradient, {"gradient_type": "fletcher_reeves"}),
])
def test_gradient_method_converges_with_correct_reason(method_fn, extra_kwargs):
    result = method_fn(**common_kwargs, **extra_kwargs)
    # These should converge for a simple quadratic
    assert result["termination_reason"] in ("gradient_norm_below_tol", "step_size_below_tol")


def test_max_iter_termination():
    # Rosenbrock function is harder to converge on
    rosenbrock = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    result = steepest_descent(f_obj=rosenbrock, x0=jnp.array([-1.0, 1.0]),
                               tol=1e-15, learning_rate=0.001, max_iter=3, verbose=False,
                               line_search_method="golden")
    assert result["termination_reason"] == "max_iter_reached"


def test_f_history_is_decreasing():
    result = steepest_descent(**common_kwargs)
    f_hist = result["f_history"]
    # For a convex quadratic, f should be monotonically decreasing
    for i in range(1, len(f_hist)):
        assert f_hist[i] <= f_hist[i - 1] + 1e-10


def test_grad_norms_decrease():
    result = steepest_descent(**common_kwargs)
    gn = result["grad_norms"]
    # Last gradient norm should be smaller than first
    assert gn[-1] < gn[0]


# --- Adaptive methods: f_history, grad_norms, termination_reason ---

@pytest.mark.parametrize("method_fn", [adam, adamax, adagrad, yogi])
def test_adaptive_method_diagnostics(method_fn):
    result = method_fn(**common_kwargs)

    assert "f_history" in result
    assert "grad_norms" in result
    assert "termination_reason" in result

    assert len(result["f_history"]) >= 1
    assert len(result["grad_norms"]) >= 1
    assert result["termination_reason"] in ("gradient_norm_below_tol", "max_iter_reached")


# --- Population/stochastic methods: f_history, termination_reason ---

@pytest.mark.parametrize("method_fn,kwargs", [
    (cmaes, {"f_obj": f_obj, "bounds": [(-5, 5), (-5, 5)], "max_iter": 20, "verbose": False}),
    (cross_entropy, {"f_obj": f_obj, "bounds": [(-5, 5), (-5, 5)], "max_iter": 20, "verbose": False}),
    (simulated_annealing, {"f_obj": f_obj, "bounds": [(-5, 5), (-5, 5)], "max_iter": 20, "verbose": False}),
    (differential_evolution, {"f_obj": f_obj, "bounds": [(-5, 5), (-5, 5)], "max_iter": 20, "verbose": False}),
    (particle_swarm, {"f_obj": f_obj, "bounds": [(-5, 5), (-5, 5)], "max_iter": 20, "verbose": False}),
])
def test_stochastic_method_diagnostics(method_fn, kwargs):
    result = method_fn(**kwargs)

    assert "f_history" in result
    assert "termination_reason" in result

    assert len(result["f_history"]) >= 1
    # f_history should be non-increasing (tracking best fitness)
    f_hist = result["f_history"]
    for i in range(1, len(f_hist)):
        assert f_hist[i] <= f_hist[i - 1] + 1e-10


def test_simulated_annealing_temperature_termination():
    # With high alpha (slow cooling), short max_iter should hit max_iter
    # With low alpha (fast cooling), should hit temperature_below_min
    result = simulated_annealing(f_obj=f_obj, bounds=[(-5, 5), (-5, 5)],
                                  max_iter=10000, T_init=1.0, T_min=1e-2, alpha=0.5,
                                  verbose=False)
    assert result["termination_reason"] == "temperature_below_min"
