
Notation and Common Definitions
================================

This section provides a summary of the notations and common definitions used throughout this guide on optimization methods.

Basic Notations
===============

Vectors and Matrices
--------------------

- **x**: Vector of decision variables.
- **b**: Vector of constants in linear constraints.
- **A**: Matrix of coefficients in linear constraints.
- **B**: Basis matrix, an :math:`m \times m` non-singular submatrix of (**A**).
- **I**: Identity matrix.

Functions and Derivatives
-------------------------
- :math:`f(x)`: Objective function to be minimized or maximized.
   - The simplest way to maximize a function is to minimize its negative. So, maximizing :math:`f(x)` is equivalent to minimizing :math:`-f(x)`.
- :math:`\nabla f(x)`: Gradient of the objective function.
- :math:`\nabla^{2} f(f)`: Hessian matrix (second-order partial derivatives) of the objective function.
- :math:`g(x)`: Constraint function.
- :math:`\nabla g(x)`: Gradient of the constraint function.

Sets and Indices
----------------
- **N**: Set of all indices.
- **B**: Set of basic variable indices.
- **N \ B**: Set of non-basic variable indices.
- **i, j, k**: Indices typically used for summation and iteration.

Common Definitions
==================

Feasible Region
---------------
The feasible region :math:`\mathcal{F}` is the set of all points that satisfy the problem's constraints:
:math:`\mathcal{F} = \{ x \in \mathbb{R}^n \mid Ax = b, \; x \geq 0 \}`

Optimal Solution
----------------
An optimal solution :math:`x^{*}` is a point in the feasible region that minimizes (or maximizes) the objective function:
:math:`x^* = \arg \min_{x \in \mathcal{F}} f(x)`

Basis and Basic Variables
--------------------------
A basis (**B**) is a set of linearly independent columns of the constraint matrix (**A**). Basic variables are those corresponding to the columns in (**B**).

Non-basic Variables
-------------------
Non-basic variables are those not in the current basis. They are typically set to zero in basic feasible solutions.

Simplex Method
--------------
An iterative method for solving linear programming problems by moving from one basic feasible solution to another, improving the objective value at each step.

Lagrange Multipliers
---------------------
Lagrange multipliers :math:`\lambda` are used to solve constrained optimization problems. They represent the sensitivity of the objective function to the constraints.

1. **Lagrangian Function**:
   :math:`L(x, \lambda) = f(x) + \lambda^T g(x)`

2. **Stationarity**:
   :math:`\nabla_x L(x^*, \lambda^*) = 0`

3. **Complementary Slackness**:
   :math:`\lambda_i g_i(x^*) = 0`

4. **Primal Feasibility**:
   :math:`g_i(x^*) \leq 0`

5. **Dual Feasibility**:

   - For minimization: :math:`\lambda^* \geq 0`
   - For maximization: :math:`\lambda^* \leq 0`

Karush-Kuhn-Tucker (KKT) Conditions
-----------------------------------
Necessary conditions for a solution in nonlinear programming to be optimal. They generalize the method of Lagrange multipliers.

1. **Stationarity**:
   :math:`\nabla f(x^*) + \sum_{i} \lambda_i \nabla g_i(x^*) = 0`

2. **Primal feasibility**:
   :math:`g_i(x^*) \leq 0`

3. **Dual feasibility**:
   :math:`\lambda_i \geq 0`

4. **Complementary slackness**:
   :math:`\lambda_i g_i(x^*) = 0`

Jacobian and Hessian
--------------------
- **Jacobian Matrix**: Matrix of all first-order partial derivatives of a vector-valued function.
- **Hessian Matrix**: Square matrix of second-order mixed partial derivatives of a scalar-valued function.

Sensitivity Analysis
--------------------
The study of how the variation in the output of a model can be apportioned to different sources of variation in the input.


