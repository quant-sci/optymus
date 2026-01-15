import jax.numpy as jnp
import pytest

from optymus.optimizer.utils.constraints import ConstraintMethodError, combine_constraints


def test_combine_constraints_flattens_and_concatenates():
    def c1(x):
        return x[0] + 1.0

    def c2(x):
        return jnp.array([x[1], x[0] - 1.0])

    combined = combine_constraints([c1, c2])
    values = combined(jnp.array([2.0, 3.0]))

    assert values.shape == (3,)
    assert jnp.allclose(values, jnp.array([3.0, 3.0, 1.0]))


def test_combine_constraints_handles_empty():
    combined = combine_constraints([])
    values = combined(jnp.array([1.0]))

    assert values.size == 0


def test_combine_constraints_rejects_non_finite_values():
    def c1(_x):
        return jnp.array([jnp.nan])

    combined = combine_constraints([c1])

    with pytest.raises(ConstraintMethodError):
        combined(jnp.array([1.0]))
