
Introduction
========================

Optimization methods are fundamental tools used across various disciplines to find the best solution to a problem. These methods span a spectrum of complexity and computational requirements, each suited to different types of optimization challenges. In this guide, we focus on methods based on continuous optimization, where the goal is to find the optimal values of continuous variables that minimize or maximize an objective function.

We can formalize the optimization problem as follows:

Given an objective function :math:`f(x)` and a set of constraints, find the values of the variables x that minimize or maximize the objective function while satisfying the constraints.

Mathematically, we can express the optimization problem as:

.. math::

    \min_{x} f(x) \quad \text{subject to} \quad g_i(x) \leq 0, \quad i = 1, 2, \ldots, m

where :math:`f(x)` is the objective function to be minimized, x is the vector of decision variables, and :math:`g_i(x)` are the constraint functions. The goal is to find the optimal values of x that minimize the objective function while satisfying the constraints.

One way to categorize optimization methods is based on the type and amount of information they use to guide the search for optimal solutions. Some methods rely on gradient information, such as first and second derivatives of the objective function, while others operate in a derivative-free manner, using only function evaluations.

In this guide, we explore a set of optimization strategies:

- Zero-order methods

    Zero-order methods, also known as derivative-free methods, are designed for scenarios where direct gradient information is unavailable or impractical to compute. These methods typically rely on function evaluations at different points in the search space to infer the optimal solution.

- First-order methods

    First-order methods utilize gradient information, such as the first derivatives of the objective function, to guide the optimization process. These methods are efficient for smooth and differentiable functions and include popular algorithms like gradient descent.

    Some popular first-order optimization algorithms include steepest descent, conjugate gradient, and BFGS (Broyden-Fletcher-Goldfarb-Shanno). BFGS is also categorized as a quasi-Newton method, which approximates the Hessian matrix using gradient information. Here we use the term "first-order" to refer to methods that use gradients, even if they approximate the Hessian matrix.

    We can write the gradient of the objective function as:

    .. math::

        \nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]

    where each component of the gradient vector represents the partial derivative of the objective function with respect to a decision variable.

- Second-order methods

    Second-order methods incorporate second-order derivatives (Hessian matrix) in addition to gradients, offering more precise information about the local curvature of the objective function. These methods can converge faster than first-order methods but require more computational resources.

    Some popular second-order optimization algorithms include Newton-Raphson method.

    We can write the Hessian matrix as:

    .. math::

        \nabla^2 f(x) = \begin{bmatrix}
            \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \ldots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
            \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \ldots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
            \vdots & \vdots & \ddots & \vdots \\
            \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \ldots & \frac{\partial^2 f}{\partial x_n^2}
        \end{bmatrix}

    where each element of the Hessian matrix represents the second partial derivative of the objective function with respect to two decision variables.

- Adaptive learning methods

    Adaptive learning methods dynamically adjust their parameters based on the progress of the optimization process or the characteristics of the problem. These methods are versatile and can improve efficiency by autonomously tuning parameters during optimization.

    Some popular adaptive learning methods include learning rate schedules, momentum, and adaptive gradient algorithms like Adam and RMSprop.

Each category addresses optimization from distinct perspectives, leveraging different types and amounts of information to guide the search for optimal solutions.

A common approach in some of this methods is the use of a line search algorithm, which is a method for finding the optimal step size along a given direction. These algorithms are essential for ensuring convergence and efficiency in optimization processes. In this guide, we explore various line search algorithms, such as the Armijo rule, Wolfe conditions, and golden section search.

Next, let's explore each optimization method in detail, starting with some notations and terminologies commonly used in optimization.