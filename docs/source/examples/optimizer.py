import sys
sys.path.append('./')

import jax.numpy as jnp
from optymus.optim import Optimizer
from optymus.plots import sphere_function

f = sphere_function()
initial_point = jnp.array([2.0, 2.0])

optimizer = Optimizer(f_obj=f,
                      x0=initial_point,
                      method='adam',
                      )

optimizer.create_dashboard(port=8041)