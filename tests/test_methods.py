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
    oc,
    adam,
    adamax,
    adagrad,
    rmsprop,
    yogi,
    cmaes,
    cross_entropy,
    simulated_annealing,
    differential_evolution,
)
from optymus import Optimizer


f_obj = lambda x: x[0] ** 2 + x[1] ** 2
x0 = jnp.array([0.0, 1.0])
tol = 1e-5
learning_rate = 0.1
max_iter = 100

def test_steepest_descent():
    result = steepest_descent(f_obj=f_obj, x0=x0, tol=tol,
                              learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter

@pytest.mark.parametrize("gradient_type", ['fletcher_reeves', 'polak_ribiere', 'hestnes_stiefel', 'dai_yuan'])
def test_conjugate_gradient(gradient_type):
    result = conjugate_gradient(f_obj=f_obj, x0=x0, tol=tol,
                                learning_rate=learning_rate, max_iter=max_iter, verbose=False, gradient_type=gradient_type)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter

def test_bfgs():
    result = bfgs(f_obj=f_obj, x0=x0, tol=tol,
                  learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter

def test_newton_raphson():
    result = newton_raphson(f_obj=f_obj, x0=x0, tol=tol,
                            learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter


def test_newton_raphson_raises_on_line_search_failure(monkeypatch):
    def failing_line_search(*_args, **_kwargs):
        raise RuntimeError("line search failed")

    monkeypatch.setattr(
        "optymus.methods.utils._base.backtracking_armijo",
        failing_line_search,
    )

    with pytest.raises(RuntimeError, match="line search failed"):
        newton_raphson(
            f_obj=f_obj,
            x0=x0,
            tol=tol,
            learning_rate=learning_rate,
            max_iter=1,
            verbose=False,
        )

def test_adam():
    result = adam(f_obj=f_obj, x0=x0, tol=tol,
                  learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < 0.01
    assert result['num_iter'] <= max_iter

def test_adamax():
    result = adamax(f_obj=f_obj, x0=x0, tol=tol,
                    learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < 0.01
    assert result['num_iter'] <= max_iter

def test_adagrad():
    result = adagrad(f_obj=f_obj, x0=x0, tol=tol,
                     learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < 0.05
    assert result['num_iter'] <= max_iter

def test_rmsprop():
    result = rmsprop(f_obj=f_obj, x0=x0, tol=tol,
                     learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert result['num_iter'] <= max_iter

def test_yogi():
    result = yogi(f_obj=f_obj, x0=x0, tol=tol,
                  learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < 0.01
    assert result['num_iter'] <= max_iter

def test_univariate():
    result = univariate(f_obj=f_obj, x0=x0, tol=tol,
                        learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter

def test_powell():
    result = powell(f_obj=f_obj, x0=x0, tol=tol,
                    learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter


def test_cmaes():
    result = cmaes(f_obj=f_obj, bounds=[(-5, 5), (-5, 5)],
                   max_iter=100, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < 0.1
    assert result['num_iter'] <= 100


def test_cross_entropy():
    result = cross_entropy(f_obj=f_obj, bounds=[(-5, 5), (-5, 5)],
                           max_iter=100, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < 0.1
    assert result['num_iter'] <= 100


def test_simulated_annealing():
    result = simulated_annealing(f_obj=f_obj, bounds=[(-5, 5), (-5, 5)],
                                  max_iter=500, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < 0.1
    assert result['num_iter'] <= 500


# --- Box constraints (variable bounds) tests ---

bounded_f_obj = lambda x: x[0] ** 2 + x[1] ** 2
bounded_x0 = jnp.array([3.0, 3.0])
bounded_bounds = [(0.5, 5.0), (0.5, 5.0)]
bounded_expected = jnp.array([0.5, 0.5])


def test_gradient_method_with_bounds():
    result = steepest_descent(f_obj=bounded_f_obj, x0=bounded_x0, bounds=bounded_bounds,
                               tol=1e-6, learning_rate=0.1, max_iter=30, verbose=False)
    assert jnp.allclose(result['xopt'], bounded_expected, atol=0.05)


def test_bfgs_with_bounds():
    result = bfgs(f_obj=bounded_f_obj, x0=bounded_x0, bounds=bounded_bounds,
                   tol=1e-6, learning_rate=0.1, max_iter=30, verbose=False)
    assert jnp.allclose(result['xopt'], bounded_expected, atol=0.05)


def test_adam_with_bounds():
    result = adam(f_obj=bounded_f_obj, x0=bounded_x0, bounds=bounded_bounds,
                   tol=1e-8, learning_rate=0.1, max_iter=50, verbose=False)
    assert jnp.allclose(result['xopt'], bounded_expected, atol=0.05)


def test_population_method_with_bounds():
    opt = Optimizer(f_obj=bounded_f_obj, x0=bounded_x0, bounds=bounded_bounds,
                     method="differential_evolution", max_iter=200, verbose=False)
    result = opt.get_results()
    assert jnp.allclose(result['xopt'], bounded_expected, atol=0.6)


def test_bounds_none_entries():
    # None means unbounded in that direction
    result = steepest_descent(f_obj=bounded_f_obj, x0=bounded_x0,
                               bounds=[(None, 5.0), (0.5, None)],
                               tol=1e-6, learning_rate=0.1, max_iter=30, verbose=False)
    xopt = result['xopt']
    assert xopt[0] < 0.05  # first dim unbounded below, should reach ~0
    assert xopt[1] >= 0.49  # second dim bounded below at 0.5


def test_bounds_backward_compat():
    # Old (lower_array, upper_array) format
    result = steepest_descent(f_obj=bounded_f_obj, x0=bounded_x0,
                               bounds=(jnp.array([0.5, 0.5]), jnp.array([5.0, 5.0])),
                               tol=1e-6, learning_rate=0.1, max_iter=30, verbose=False)
    assert jnp.allclose(result['xopt'], bounded_expected, atol=0.05)


# --- Line search integration tests ---

def test_steepest_descent_armijo():
    result = steepest_descent(f_obj=f_obj, x0=x0, tol=tol,
                              max_iter=max_iter, verbose=False,
                              line_search_method="armijo")
    assert jnp.linalg.norm(result['xopt']) < tol


def test_bfgs_wolfe():
    result = bfgs(f_obj=f_obj, x0=x0, tol=tol,
                  max_iter=max_iter, verbose=False,
                  line_search_method="wolfe")
    assert jnp.linalg.norm(result['xopt']) < tol


def test_bfgs_golden_override():
    result = bfgs(f_obj=f_obj, x0=x0, tol=tol,
                  learning_rate=learning_rate, max_iter=max_iter, verbose=False,
                  line_search_method="golden")
    assert jnp.linalg.norm(result['xopt']) < tol


def test_lbfgs():
    result = lbfgs(f_obj=f_obj, x0=x0, tol=tol,
                   max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol


# --- Optimality Criteria (OC) tests ---

oc_f_obj = lambda x: jnp.sum(x**2)
oc_g_con = lambda x: 1.0 - jnp.sum(x)  # sum(x) >= 1
oc_x0 = jnp.array([0.5, 0.5])
oc_bounds = [(0.01, 2.0), (0.01, 2.0)]


def test_oc():
    result = oc(f_obj=oc_f_obj, f_cons=[oc_g_con], x0=oc_x0, bounds=oc_bounds,
                tol=1e-4, max_iter=200, verbose=False)
    # Optimal: x = [0.5, 0.5] with sum(x) = 1
    assert jnp.allclose(result['xopt'], jnp.array([0.5, 0.5]), atol=0.01)
    assert result['termination_reason'] in ('density_change_below_tol', 'max_iter_reached')


def test_oc_requires_bounds():
    with pytest.raises(ValueError, match="requires bounds"):
        oc(f_obj=oc_f_obj, f_cons=[oc_g_con], x0=oc_x0,
           tol=1e-4, max_iter=10, verbose=False)


def test_oc_requires_f_cons():
    with pytest.raises(ValueError, match="requires.*constraint"):
        oc(f_obj=oc_f_obj, x0=oc_x0, bounds=oc_bounds,
           tol=1e-4, max_iter=10, verbose=False)


def test_oc_via_optimizer():
    opt = Optimizer(f_obj=oc_f_obj, f_cons=[oc_g_con], x0=oc_x0,
                    method="oc", bounds=oc_bounds, max_iter=100, tol=1e-4, verbose=False)
    result = opt.get_results()
    assert 'xopt' in result
    assert result['termination_reason'] in ('density_change_below_tol', 'max_iter_reached')


def test_oc_high_dimension():
    n = 10
    result = oc(f_obj=lambda x: jnp.sum(x**2),
                f_cons=[lambda x: 1.0 - jnp.sum(x)],
                x0=jnp.ones(n) * 0.5,
                bounds=[(0.01, 2.0)] * n,
                tol=1e-4, max_iter=300, verbose=False)
    # Optimal: x_i = 1/n = 0.1
    assert jnp.allclose(result['xopt'], jnp.ones(n) * 0.1, atol=0.02)
