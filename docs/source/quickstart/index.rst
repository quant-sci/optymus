Quickstart
===================================

This is a quickstart guide to get you up and running with the `optymus` project.


To begin using `optymus`, follow these steps:

1. **Install the package via PyPI:**


.. code-block:: bash

   pip install optymus

2. **Create an optimization problem:**

.. code-block:: python

   from optymus import Optimizer
   from optymus.benchmark import MccormickFunction
   
   import jax.numpy as jnp

   f = MccormickFunction()
   initial_point = jnp.array([2, 2])

3. **Optimize the problem:**

.. code-block:: python

   opt = Optimizer(f_obj=f,
                   x0=initial_point,
                   method='bfgs')

4. **Print the optimization report:**

.. code-block:: python

   opt.print_report()

5. **Visualize the optimization process:**

.. code-block:: python

   opt.plot_results()

.. toctree::
   :maxdepth: 3