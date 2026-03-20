import jax
import jax.numpy as jnp

from optymus.search import backtracking_armijo, bracket_minimum, line_search, wolfe_line_search


def test_backtracking_armijo_quadratic():
    f = lambda x: jnp.dot(x, x)
    x = jnp.array([1.0, 2.0])
    grad = jax.grad(f)(x)
    d = -grad  # descent direction

    result = backtracking_armijo(f=f, x=x, d=d, grad=grad)
    alpha = result["alpha"]

    # Verify Armijo condition holds
    c1 = 1e-4
    assert f(x + alpha * d) <= f(x) + c1 * alpha * jnp.dot(grad, d)
    # Verify we actually decreased the function
    assert result["fmin"] < f(x)


def test_backtracking_armijo_terminates():
    # Adversarial: flat function where Armijo is trivially satisfied
    f = lambda x: jnp.float32(0.0)
    x = jnp.array([1.0, 1.0])
    grad = jnp.array([0.0, 0.0])
    d = jnp.array([-1.0, -1.0])

    result = backtracking_armijo(f=f, x=x, d=d, grad=grad, max_iter=10)
    assert result["alpha"] > 0
    assert result["num_iter"] <= 10


def test_wolfe_quadratic():
    f = lambda x: jnp.dot(x, x)
    grad_f = jax.grad(f)
    x = jnp.array([3.0, 4.0])
    grad = grad_f(x)
    d = -grad  # descent direction

    result = wolfe_line_search(f=f, grad_f=grad_f, x=x, d=d, grad=grad)
    alpha = result["alpha"]

    c1 = 1e-4
    c2 = 0.9
    slope0 = jnp.dot(grad, d)

    # Sufficient decrease (Armijo)
    assert f(x + alpha * d) <= f(x) + c1 * alpha * slope0
    # Curvature condition (strong Wolfe)
    assert jnp.abs(jnp.dot(grad_f(x + alpha * d), d)) <= -c2 * slope0
    # Verify decrease
    assert result["fmin"] < f(x)


def test_bracket_minimum_max_iter():
    # Function that never triggers the bracket condition within a few steps
    f = lambda x: -jnp.dot(x, x)  # concave — always decreasing along any direction from origin
    x0 = jnp.array([0.0, 0.0])
    d = jnp.array([1.0, 0.0])

    # Should not hang — returns best bracket after max_iter
    a, b = bracket_minimum(func=f, x0=x0, dir=d, max_iter=10)
    assert jnp.isfinite(a) and jnp.isfinite(b)


def test_golden_tol_configurable():
    f = lambda x: jnp.dot(x, x)
    x = jnp.array([1.0, 1.0])
    d = jnp.array([-1.0, -1.0])

    r1 = line_search(f=f, x=x, d=d, learning_rate=0.01, tol=1e-3)
    r2 = line_search(f=f, x=x, d=d, learning_rate=0.01, tol=1e-8)

    # Both should find the minimum, tighter tol should be more precise
    assert r1["fmin"] < f(x)
    assert r2["fmin"] < f(x)
    # Tighter tolerance should produce a result at least as good
    assert r2["fmin"] <= r1["fmin"] + 1e-6
