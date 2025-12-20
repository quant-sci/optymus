<h1 align="center">
<img src="logo.svg" width="400">
</h1>

[![PyPI](https://img.shields.io/pypi/v/optymus)](https://pypi.org/project/optymus/)
[![GitHub](https://img.shields.io/github/license/kleyt0n/optymus)](https://github.com/kleyt0n/optymus/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/optymus/badge/?version=latest)](https://optymus.readthedocs.io/en/latest/?badge=latest)
![jax_badge][jax_badge_link]

**Structural Optimization and Scientific Computing**

`optymus` is a Python library for solving optimization problems in mechanical engineering and scientific computing. Built on [JAX](https://jax.readthedocs.io/en/latest/index.html) for automatic differentiation, it provides efficient gradient computation and GPU acceleration. The library is designed for structural optimization, topology optimization, and general-purpose numerical optimization.

## Key Features

- **Topology Optimization**: PolyMesher-based mesh generation with signed distance functions
- **Pre-built Engineering Domains**: Cook membrane, Michell truss, MBB beam, Wrench, Suspension
- **Structural Analysis Support**: Boundary conditions, load definitions, and FEM-ready domains
- **Multiple Optimization Methods**: Gradient-based (BFGS, Conjugate Gradient, Newton), adaptive (Adam, AdaGrad), and evolutionary (CMA-ES, Differential Evolution, Particle Swarm)
- **Constraint Handling**: Support for equality and inequality constraints via penalty methods
- **Visualization**: Built-in plotting for optimization paths, convergence, and domain visualization

## Getting Started

1. Install optymus:
   ```bash
   uv add optymus
   ```

2. Basic optimization example:
   ```python
   from optymus import Optimizer
   from optymus.benchmark import Mccormick

   import jax.numpy as jnp

   f = Mccormick()
   initial_point = jnp.array([2.0, 2.0])

   opt = Optimizer(f_obj=f,
                   x0=initial_point,
                   method='bfgs')

   opt.report()
   ```

3. Topology optimization with engineering domains:
   ```python
   from optymus.benchmark import MbbDomain
   from optymus.methods import polymesher

   # MBB beam domain with boundary conditions
   domain = MbbDomain

   # Generate polygonal mesh
   result = polymesher(domain=domain, num_elements=100)
   ```

## Documentation

Visit the [official documentation](https://optymus.readthedocs.io/en/latest/) for:
- Complete API reference
- Optimization method guides
- Mechanical engineering examples
- Topology optimization tutorials

## Citation

If you use `optymus` in your research, please cite:

```bibtex
@misc{optymus2024,
  author = {da Costa, Kleyton and Menezes, Ivan and Lopes, Helio},
  title = {Optymus: Optimization Methods in Python},
  year = {2024},
  note = {GitHub Repository},
  url = {https://github.com/quant-sci/optymus}
}
```


[jax_link]: https://github.com/google/jax
[jax_badge_link]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,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
