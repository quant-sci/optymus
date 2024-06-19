.. optymus documentation master file, created by
   sphinx-quickstart on Tue Jan  2 19:22:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

optymus documentation
===================================


.. image:: https://img.shields.io/pypi/v/optymus
.. image:: https://img.shields.io/github/license/quant-sci/optymus
.. image:: https://readthedocs.org/projects/optymus/badge/?version=latest
.. image:: https://img.shields.io/pypi/dm/optymus.svg?label=PyPI%20downloads


This library provides a comprehensive collection of optimization methods, both with and without constraints. The main goal is provide a simple structure to improve research and development in optimization problems.

Features
--------

.. dropdown:: Optimization Methods
   :icon: rocket

   **Zero-order**
      - Univariant
      - Powell's method

   **First-order**
      - Gradient Descent
      - Conjugate Gradient
      - BFGS

   **Second-order**
      - Newton's method
   
   **Adaptative learning**
      - Adam
      - Adamax
      - RMSprop
      - Adagrad
      - Yogi


.. dropdown:: Benchmark Functions
   :icon: ellipsis

   We provide a set of benchmark functions to test the optimization methods. The following functions are available:

      - Ackley
      - Beale
      - Booth
      - CrossInTray
      - Easom
      - Eggholder
      - GoldsteinPrice
      - Mccormick
      - Rastrigin
      - Rosenbrock
      - StyblinskiTang

   **Obs.:** You can also create your own benchmark function by inheriting the `CustomFunction` class.

.. dropdown:: Visualization
   :icon: eye

   **Plotting**
      - Plot the optimization path
      - Counter plot of the function

   **Reports**
      - Create a report with the optimization results


Getting Started
---------------

To begin using `optymus`, follow these steps:

1. **Install the package:**


.. code-block:: bash

   pip install optymus

2. **Create an optimization problem:**

.. code-block:: python

   from optymus import Optimizer
   from optymus.benchmark import MccormickFunction
   
   import numpy as np

   f = MccormickFunction()
   initial_point = np.array([2, 2])

3. **Optimize the problem:**

.. code-block:: python

   opt = Optimizer(f_obj=f,
                   x0=initial_point,
                   method='bfgs')

4. **Print the optimization report:**

.. code-block:: python

   opt.print_report()

Contributions
-------------

Contributions to Optymus are highly appreciated. If you have additional optimization methods, improvements, or bug fixes, please submit a pull request following the CONTRIBUTING.md on GitHub.

How to cite
-------------

If you use `optymus` in your research, please consider citing the library using the following BibTeX entry:

.. code-block:: bibtex
   
   @misc{optymus2024,
   author = {da Costa, Kleyton and Menezes, Ivan and Lopes, Helio},
   title = {Optymus: Optimization Methods Library for Python},
   year = {2024},
   note = {GitHub Repository},
   url = {https://github.com/quant-sci/optymus}
   }

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:

   quickstart/index
   reference/index
   examples/index

.. note::
   
   optymus is part of `quantsci <https://quantsci.org>`_ project.

