# Roadmap to v0.2.0

**Current version:** 0.1.11
**Target version:** 0.2.0

---

## 1. Core Optimization v0.1.2

- [x] Fix version mismatch (`pyproject.toml` is 0.1.11, `__about__.py` is 0.1.10)
- [x] Add box constraints (variable bounds) support
- [ ] Improve line search integration across all gradient methods (steepest descent, BFGS, L-BFGS, conjugate gradient)
- [ ] Add convergence diagnostics: iteration history, termination reasons, gradient norms
- [ ] Implement Optimality Criteria (OC) method (stub exists at `src/optymus/methods/zero_order/_oc.py`)
- [ ] Unified result object with convergence info across all method families

## 2. FEM & Engineering v0.1.3

- [ ] Implement OC method for topology optimization
- [ ] Add stress and displacement constraints for topology problems
- [ ] Expand topological domains library (beyond PolyMesher)
- [ ] Add 3D FEM support (plane strain / solid elements)

## 3. Testing v0.1.4

- [ ] Expand test coverage (currently only `test_methods`, `test_constraint_methods`, `test_constraints_utils`)
- [ ] Add FEM / topology optimization integration tests
- [ ] Add benchmark function tests across all 16 functions in `optymus.benchmark`
- [ ] Add tests for all method families: adaptive, first-order, second-order, zero-order, population, stochastic
- [ ] Add performance regression tests

## 4. Documentation & Examples v0.1.5

- [ ] Add constraint optimization guide (penalty & barrier methods)
- [ ] Add FEM / topology optimization tutorial
- [ ] Add method selection guide (decision tree for choosing the right solver)
- [ ] Create example notebooks for common workflows

## 5. Visualization (new) v0.1.6

- [ ] Design lightweight plotting utilities (replace the removed `optymus.plots` module)
- [ ] Convergence history plots
- [ ] Mesh / topology result visualization for FEM
- [ ] Choose plotting backend (`matplotlib` as optional dependency)
