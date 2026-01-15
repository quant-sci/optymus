
Optimizer
=========

optymus provides a class to optimize a function using a selected optimization algorithm.

This function can be called using `from optymus import Optimizer` class.



Optimizer
------------------

.. autosummary::
    :toctree: .generated/

    optymus.Optimizer


Constraint Methods
------------------

`Optimizer` supports indirect constraint handling with penalty and barrier
wrappers. Inequality constraints should be provided as `g(x) <= 0` and
equalities as `h(x) == 0`.

Example:

.. code-block:: python

    import jax.numpy as jnp
    from optymus import Optimizer

    def f_obj(x):
        return (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2

    def g1(x):
        return x[0] + x[1] + 3.0

    def h1(x):
        return x[0] - x[1]

    opt = Optimizer(
        f_obj=f_obj,
        g_cons=[g1],
        h_cons=[h1],
        x0=jnp.array([-5.0, -2.0]),
        method="bfgs",
        constraint_method="penalty",
        constraint_jit=False,
        penalty_r0=1.0,
        penalty_factor=0.1,
        max_outer_iter=6,
        verbose=False,
    )

The barrier method uses a log barrier by default; set `barrier_type="inverse"`
to switch to an inverse barrier.

Set `constraint_jit=True` to JIT-compile the constrained objective when the
objective and constraints use `jax.numpy` operations.
