<h1 align="center">
<img src="logo.svg" width="400">
</h1>

[![PyPI](https://img.shields.io/pypi/v/optymus)](https://pypi.org/project/optymus/)
[![GitHub](https://img.shields.io/github/license/kleyt0n/optymus)](https://github.com/kleyt0n/optymus/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/optymus/badge/?version=latest)](https://optymus.readthedocs.io/en/latest/?badge=latest)

> optymus is part of [quantsci](https://quantsci.org) project.

This library provides a comprehensive collection of optimization methods, both with and without constraints. The main goal is provide a simple structure to improve research and development in optimization problems.


## Getting Started

To begin using `optymus`, follow these steps:

1. **Install optymus:**
   ```bash
   pip install optymus
   ```

2. **Get Started:**
   ```python
   from optymus import Optimizer
   from optymus.benchmark import MccormickFunction
   
   import numpy as np

   f = MccormickFunction()
   initial_point = np.array([2, 2])

   opt = Optimizer(f_obj=f,
                   x0=initial_point,
                   method='bfgs')

   opt.report()
   ```

3. **Explore the Documentation:**
   Visit the [official documentation](https://optymus-docs.readthedocs.com) to understand the available optimization methods and how to use them effectively.

Refer to the documentation for detailed information on each method and its application.

## Contributions

Contributions to Optymus are highly appreciated. If you have additional optimization methods, improvements, or bug fixes, please submit a pull request following the [contribution guidelines](CONTRIBUTING.md).

## Cite

If you use `optymus` in your research, please consider citing the library using the following BibTeX entry:

```bibtex
@misc{optymus2024,
  author = {da Costa, Kleyton and Menezes, Ivan and Lopes, Helio},
  title = {Optymus: Optimization Methods Library for Python},
  year = {2024},
  note = {GitHub Repository},
  url = {https://github.com/quant-sci/optymus}
}
```