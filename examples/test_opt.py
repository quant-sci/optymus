import sys
sys.path.append('./')

import jax.numpy as jnp
from optymus.optim import Optimizer
from optymus.utils import mccormick_function

f = mccormick_function()
initial_point = jnp.array([2.0, 2.0])

optimizer = Optimizer(f_obj=f,
                      x0=initial_point,
                      method='bfgs')