import pytest
import jax.numpy as jnp

from optymus import Optimizer
from optymus.optimizer._optimizer import ConstraintMethodError, InfeasibleStartError
from optymus.optimizer.utils.constraints import run_barrier_method, run_penalty_method


def quadratic_obj(x):
    return (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2


def inequality_constraint(x):
    return x[0] + x[1] + 3.0


def equality_constraint(x):
    return x[0] - x[1]


def nan_constraint(_x):
    return jnp.array([jnp.nan])


def vector_inequality_constraint(x):
    return jnp.array([x[0] + x[1] + 3.0, x[0] + x[1] + 3.0])


def nan_objective(_x):
    return jnp.array(jnp.nan)


def equality_only_objective(x):
    return (x[0] + 1.0) ** 2 + (x[1] + 1.0) ** 2


def test_penalty_method_handles_inequality_and_equality():
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="penalty",
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert inequality_constraint(xopt) <= 1e-2
    assert jnp.abs(equality_constraint(xopt)) <= 1e-2
    assert jnp.linalg.norm(xopt - jnp.array([-1.5, -1.5])) < 0.2


def test_penalty_method_default_schedule_increases():
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="steepest_descent",
        constraint_method="penalty",
        max_outer_iter=2,
        max_iter=0,
        learning_rate=0.05,
        constraint_tol=-1.0,
        outer_tol=-1.0,
        verbose=False,
    )
    result = opt.get_results()
    r_p_history = result["r_p_history"]

    assert float(r_p_history[1]) > float(r_p_history[0])


@pytest.mark.parametrize("barrier_type", ["log", "inverse"])
def test_barrier_method_supports_log_and_inverse_barriers(barrier_type):
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="barrier",
        barrier_type=barrier_type,
        barrier_r0=10.0,
        barrier_factor=0.1,
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert inequality_constraint(xopt) <= 1e-2
    assert jnp.abs(equality_constraint(xopt)) <= 1e-2


def test_barrier_method_requires_strictly_feasible_start():
    with pytest.raises(InfeasibleStartError):
        Optimizer(
            f_obj=quadratic_obj,
            g_cons=[inequality_constraint],
            h_cons=[equality_constraint],
            x0=jnp.array([2.0, 2.0]),
            method="bfgs",
            constraint_method="barrier",
            barrier_r0=10.0,
            barrier_factor=0.1,
            penalty_r0=1.0,
            penalty_factor=10.0,
            max_outer_iter=3,
            max_iter=100,
            learning_rate=0.1,
            verbose=False,
        )


def test_barrier_method_rejects_boundary_start():
    with pytest.raises(InfeasibleStartError):
        Optimizer(
            f_obj=quadratic_obj,
            g_cons=[inequality_constraint],
            h_cons=[equality_constraint],
            x0=jnp.array([-2.0, -1.0]),
            method="bfgs",
            constraint_method="barrier",
            barrier_r0=10.0,
            barrier_factor=0.1,
            penalty_r0=1.0,
            penalty_factor=10.0,
            max_outer_iter=3,
            max_iter=100,
            learning_rate=0.1,
            verbose=False,
        )


def test_barrier_method_default_penalty_schedule_increases():
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="steepest_descent",
        constraint_method="barrier",
        max_outer_iter=2,
        max_iter=0,
        learning_rate=0.05,
        constraint_tol=-1.0,
        outer_tol=-1.0,
        verbose=False,
    )
    result = opt.get_results()
    r_p_history = result["r_p_history"]

    assert float(r_p_history[1]) > float(r_p_history[0])


def test_barrier_method_backtracks_infeasible_iterate():
    def inner_method(f_obj, f_cons, x0, **_kwargs):
        return {"xopt": jnp.array([2.0, 2.0]), "num_iter": 1}

    result = run_barrier_method(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        inner_method=inner_method,
        constraint_jit=False,
        barrier_type="log",
        barrier_r0=1.0,
        barrier_factor=0.1,
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=1,
        constraint_tol=-1.0,
        outer_tol=-1.0,
        barrier_eps=1e-12,
        warn_constraint_size=None,
        warn_slow_iter_s=None,
        warn_no_progress_iters=None,
        warn_no_progress_tol=1e-6,
        inner_kwargs={},
    )

    assert inequality_constraint(result["xopt"]) < 0.0
    assert jnp.isclose(result["fmin"], quadratic_obj(result["xopt"]))


def test_barrier_method_allows_equality_only():
    opt = Optimizer(
        f_obj=quadratic_obj,
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="barrier",
        barrier_r0=10.0,
        barrier_factor=0.1,
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert jnp.abs(equality_constraint(xopt)) <= 1e-2


@pytest.mark.parametrize(
    ("method", "method_kwargs"),
    [
        ("bfgs", {}),
        ("steepest_descent", {}),
    ],
)
def test_penalty_method_supports_multiple_methods(method, method_kwargs):
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method=method,
        constraint_method="penalty",
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
        **method_kwargs,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert inequality_constraint(xopt) <= 1e-2
    assert jnp.abs(equality_constraint(xopt)) <= 1e-2


@pytest.mark.parametrize(
    ("method", "method_kwargs"),
    [
        ("bfgs", {}),
        ("steepest_descent", {}),
    ],
)
def test_barrier_method_supports_multiple_methods(method, method_kwargs):
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method=method,
        constraint_method="barrier",
        barrier_r0=10.0,
        barrier_factor=0.1,
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
        **method_kwargs,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert inequality_constraint(xopt) <= 1e-2
    assert jnp.abs(equality_constraint(xopt)) <= 1e-2


def test_penalty_method_avoids_duplicate_objective_eval():
    counter = {"count": 0}

    def f_obj(x):
        counter["count"] += 1
        return jnp.sum(x**2)

    def inner_method(f_obj, f_cons, x0, **_kwargs):
        return {"xopt": x0 + jnp.ones_like(x0), "num_iter": 1}

    run_penalty_method(
        f_obj=f_obj,
        g_cons=[],
        h_cons=[],
        x0=jnp.array([0.0, 1.0]),
        inner_method=inner_method,
        constraint_jit=False,
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=2,
        constraint_tol=-1.0,
        outer_tol=-1.0,
        warn_constraint_size=None,
        warn_slow_iter_s=None,
        warn_no_progress_iters=None,
        warn_no_progress_tol=1e-6,
        inner_kwargs={},
    )

    assert counter["count"] == 2


def test_barrier_method_avoids_duplicate_objective_eval():
    counter = {"count": 0}

    def f_obj(x):
        counter["count"] += 1
        return jnp.sum(x**2)

    def inner_method(f_obj, f_cons, x0, **_kwargs):
        return {"xopt": x0 + jnp.ones_like(x0), "num_iter": 1}

    run_barrier_method(
        f_obj=f_obj,
        g_cons=[],
        h_cons=[],
        x0=jnp.array([0.0, 1.0]),
        inner_method=inner_method,
        constraint_jit=False,
        barrier_type="log",
        barrier_r0=1.0,
        barrier_factor=0.1,
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=2,
        constraint_tol=-1.0,
        outer_tol=-1.0,
        barrier_eps=1e-12,
        warn_constraint_size=None,
        warn_slow_iter_s=None,
        warn_no_progress_iters=None,
        warn_no_progress_tol=1e-6,
        inner_kwargs={},
    )

    assert counter["count"] == 2


def test_penalty_method_rejects_non_finite_constraints():
    with pytest.raises(ConstraintMethodError):
        Optimizer(
            f_obj=quadratic_obj,
            g_cons=[nan_constraint],
            x0=jnp.array([-5.0, -2.0]),
            method="adam",
            constraint_method="penalty",
            penalty_r0=1.0,
            penalty_factor=10.0,
            max_outer_iter=2,
            max_iter=50,
            learning_rate=0.05,
            constraint_tol=1e-3,
            outer_tol=1e-3,
            verbose=False,
        )


def test_penalty_method_handles_vector_constraints():
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[vector_inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="penalty",
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert vector_inequality_constraint(xopt).max() <= 1e-2
    assert jnp.abs(equality_constraint(xopt)) <= 1e-2


def test_penalty_method_bails_on_nan_objective():
    with pytest.raises(ConstraintMethodError):
        Optimizer(
            f_obj=nan_objective,
            g_cons=[inequality_constraint],
            h_cons=[equality_constraint],
            x0=jnp.array([-5.0, -2.0]),
            method="adam",
            constraint_method="penalty",
            penalty_r0=1.0,
            penalty_factor=10.0,
            max_outer_iter=2,
            max_iter=50,
            learning_rate=0.05,
            constraint_tol=1e-3,
            outer_tol=1e-3,
            verbose=False,
        )


def test_barrier_method_bails_on_nan_objective():
    with pytest.raises(ConstraintMethodError):
        Optimizer(
            f_obj=nan_objective,
            g_cons=[inequality_constraint],
            h_cons=[equality_constraint],
            x0=jnp.array([-5.0, -2.0]),
            method="adam",
            constraint_method="barrier",
            barrier_r0=10.0,
            barrier_factor=0.1,
            penalty_r0=1.0,
            penalty_factor=10.0,
            max_outer_iter=2,
            max_iter=50,
            learning_rate=0.05,
            constraint_tol=1e-3,
            outer_tol=1e-3,
            verbose=False,
        )


def test_penalty_method_handles_infeasible_bounds():
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([0.5, -0.5]),
        method="adam",
        constraint_method="penalty",
        penalty_r0=1.0,
        penalty_factor=10.0,
        bounds=(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert inequality_constraint(xopt) > 0.0


def test_penalty_method_handles_equality_only():
    opt = Optimizer(
        f_obj=equality_only_objective,
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="penalty",
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert jnp.abs(equality_constraint(xopt)) <= 1e-2


@pytest.mark.parametrize("penalty_factor", [2.0, 5.0, 10.0])
def test_penalty_method_equality_only_schedule_stability(penalty_factor):
    opt = Optimizer(
        f_obj=equality_only_objective,
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="penalty",
        penalty_r0=1.0,
        penalty_factor=penalty_factor,
        max_outer_iter=4,
        max_iter=150,
        learning_rate=0.05,
        constraint_tol=1e-3,
        outer_tol=1e-3,
        verbose=False,
    )
    result = opt.get_results()
    xopt = result["xopt"]

    assert jnp.abs(equality_constraint(xopt)) <= 1e-2


def test_penalty_method_emits_guardrail_warnings():
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[vector_inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="penalty",
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=2,
        max_iter=50,
        learning_rate=0.05,
        constraint_tol=0.0,
        outer_tol=0.0,
        warn_constraint_size=1,
        warn_slow_iter_s=0.0,
        warn_no_progress_iters=1,
        warn_no_progress_tol=1e9,
        verbose=False,
    )
    result = opt.get_results()
    warnings = set(result.get("warnings", []))

    assert "constraint_size_warning" in warnings
    assert "slow_iteration_warning" in warnings
    assert "no_progress_warning" in warnings
    assert "max_outer_iter_reached" in warnings


def test_barrier_method_emits_guardrail_warnings():
    opt = Optimizer(
        f_obj=quadratic_obj,
        g_cons=[vector_inequality_constraint],
        h_cons=[equality_constraint],
        x0=jnp.array([-5.0, -2.0]),
        method="adam",
        constraint_method="barrier",
        barrier_r0=10.0,
        barrier_factor=0.1,
        penalty_r0=1.0,
        penalty_factor=10.0,
        max_outer_iter=2,
        max_iter=50,
        learning_rate=0.05,
        constraint_tol=0.0,
        outer_tol=0.0,
        warn_constraint_size=1,
        warn_slow_iter_s=0.0,
        warn_no_progress_iters=1,
        warn_no_progress_tol=1e9,
        verbose=False,
    )
    result = opt.get_results()
    warnings = set(result.get("warnings", []))

    assert "constraint_size_warning" in warnings
    assert "slow_iteration_warning" in warnings
    assert "no_progress_warning" in warnings
    assert "max_outer_iter_reached" in warnings
