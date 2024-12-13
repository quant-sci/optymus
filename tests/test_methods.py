import pytest
import jax.numpy as jnp
from optymus.methods import (
    steepest_descent, 
    conjugate_gradient, 
    bfgs, 
    newton_raphson, 
    univariate,
    powell,
    adam, 
    adamax,
    adagrad, 
    rmsprop,
    yogi,
)


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

def test_adam():
    result = adam(f_obj=f_obj, x0=x0, tol=tol,
                  learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter

def test_adamax():
    result = adamax(f_obj=f_obj, x0=x0, tol=tol,
                    learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter

def test_adagrad():
    result = adagrad(f_obj=f_obj, x0=x0, tol=tol,
                     learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
    assert result['num_iter'] <= max_iter

def test_rmsprop():
    result = rmsprop(f_obj=f_obj, x0=x0, tol=tol,
                     learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert result['num_iter'] <= max_iter

def test_yogi():
    result = yogi(f_obj=f_obj, x0=x0, tol=tol,
                  learning_rate=learning_rate, max_iter=max_iter, verbose=False)
    assert jnp.linalg.norm(result['xopt']) < tol
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