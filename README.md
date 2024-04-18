<h1 align="center">
<img src="logo.svg" width="400">
</h1>

[![PyPI](https://img.shields.io/pypi/v/optymus)](https://pypi.org/project/optymus/)
[![GitHub](https://img.shields.io/github/license/kleyt0n/optymus)](https://github.com/kleyt0n/optymus/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/optymus/badge/?version=latest)](https://optymus.readthedocs.io/en/latest/?badge=latest)

> Optymus is part of [quantsci](https://quantsci.org) project.

This library provides a comprehensive collection of optimization methods, both with and without constraints. The main goal is provide a simple structure to improve research and development in optimization problems.

## Implemented Methods

| Method | Description |
| --- | --- |
| bfgs | Broyden-Fletcher-Goldfarb-Shanno (BFGS) |
| steepdesc | Steepest Descent |
| newton_raphson | Newton-Raphson Method |
| powell | Powell's Method |
|fletcher_reeves | Fletcher-Reeves |


## Getting Started

To begin using _optymus_, follow these steps:

1. **Install optymus:**
   ```bash
   pip install optymus
   ```

2. **Explore the Documentation:**
   Visit the [official documentation](https://optymus-docs.readthedocs.com) to understand the available optimization methods and how to use them effectively.

3. **Get Started:**
   ```python
   from optymus.optim import Optimizer
   from optymus.utils import mccormick_function
   
   import numpy as np

   f = mccormick_function()
   initial_point = np.array([2, 2])

   opt = Optimizer(f_obj=f,
                   x0=initial_point,
                   method='bfgs')

   opt.report()

   opt.plot()
   ```

Refer to the documentation for detailed information on each method and its application.

## Implement your own method an compare with the implemented ones

We are working to implement a simple way to add your own optimization method. 

## Contributions

Contributions to Optymus are highly appreciated. If you have additional optimization methods, improvements, or bug fixes, please submit a pull request following the [contribution guidelines](CONTRIBUTING.md).

## Cite

If you use Optymus in your research, please consider citing the library using the following BibTeX entry:

```bibtex
@misc{optymus2024,
  author = {Costa, Kleyton and Menezes, Ivan},
  title = {Optymus: Optimization Methods Library for Python},
  year = {2024},
  note = {GitHub Repository},
  url = {https://github.com/quantsci/optymus}
}
```