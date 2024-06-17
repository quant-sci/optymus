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


Explore the Documentation
-------------------------
Visit the `official documentation <https://optymus.readthedocs.io/en/latest/?badge=latest>`_ to understand the available optimization methods and how to use them effectively.

Refer to the documentation for detailed information on each method and its application.

Contributions
-------------

Contributions to Optymus are highly appreciated. If you have additional optimization methods, improvements, or bug fixes, please submit a pull request following the [contribution guidelines](CONTRIBUTING.md).

How to cite
----

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

