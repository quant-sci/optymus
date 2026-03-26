import jax.numpy as jnp

from optymus import OptimizeResult
from optymus.methods import steepest_descent, particle_swarm


def _make_result(**overrides):
    base = {
        "method_name": "Test",
        "xopt": jnp.array([1.0, 2.0]),
        "fmin": 0.5,
        "num_iter": 10,
        "termination_reason": "gradient_norm_below_tol",
        "time": 0.1234,
    }
    base.update(overrides)
    return OptimizeResult(base)


# --- Dict backward compatibility ---

def test_dict_access():
    r = _make_result()
    assert r["xopt"] is not None
    assert r.get("fmin") == 0.5
    assert "num_iter" in r
    assert r.get("nonexistent", 42) == 42


def test_attribute_access():
    r = _make_result()
    assert r.xopt is r["xopt"]
    assert r.fmin == r["fmin"]
    assert r.method_name == "Test"


def test_attribute_set_and_delete():
    r = _make_result()
    r.new_field = 99
    assert r["new_field"] == 99
    del r.new_field
    assert "new_field" not in r


# --- Converged property ---

def test_converged_true():
    for reason in [
        "gradient_norm_below_tol",
        "step_size_below_tol",
        "density_change_below_tol",
        "temperature_below_min",
        "std_below_min",
        "constraint_converged",
    ]:
        r = _make_result(termination_reason=reason)
        assert r.converged is True, f"Expected converged=True for {reason}"


def test_converged_false():
    r = _make_result(termination_reason="max_iter_reached")
    assert r.converged is False


def test_converged_missing_reason():
    r = OptimizeResult({"xopt": jnp.array([0.0])})
    assert r.converged is False


# --- Repr ---

def test_repr_no_error():
    r = _make_result()
    text = repr(r)
    assert "Test" in text
    assert "Converged" in text
    assert "True" in text


def test_repr_mimebundle():
    r = _make_result()
    bundle = r._repr_mimebundle_()
    assert "text/plain" in bundle


# --- Integration with actual methods ---

def test_steepest_descent_returns_optimize_result():
    f_obj = lambda x: x[0] ** 2 + x[1] ** 2
    result = steepest_descent(f_obj=f_obj, x0=jnp.array([1.0, 1.0]),
                               tol=1e-5, max_iter=50, verbose=False)
    assert isinstance(result, OptimizeResult)
    assert result.converged is True
    assert result.xopt is result["xopt"]
    assert result["termination_reason"] in ("gradient_norm_below_tol", "step_size_below_tol")
    text = repr(result)
    assert "Steepest Descent" in text


def test_population_method_returns_optimize_result():
    f_obj = lambda x: x[0] ** 2 + x[1] ** 2
    result = particle_swarm(f_obj=f_obj, bounds=[(-5, 5), (-5, 5)],
                             max_iter=5, verbose=False)
    assert isinstance(result, OptimizeResult)
    assert result.converged is False  # PSO always max_iter_reached
    assert result.termination_reason == "max_iter_reached"
