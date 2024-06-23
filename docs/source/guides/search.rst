
Line Search Methods
===================

Line search methods are techniques used in optimization to find an appropriate step size along a given search direction. These methods aim to efficiently determine the step size that minimizes the objective function. This section covers some common line search methods.

Golden Section Search
---------------------

The Golden Section Search is an iterative method used to find the minimum of a unimodal function. It is based on the golden ratio and reduces the interval of uncertainty at each iteration.

Algorithm:

1. Choose an initial interval :math:`[a, b]` such that the minimum lies within this interval.

2. Compute the interior points using the golden ratio: 

   .. math:: 

      c = a + \frac{b - a}{\phi}, \quad d = b - \frac{b - a}{\phi}

   where :math:`\phi = \frac{1 + \sqrt{5}}{2}` is the golden ratio.

3. Evaluate the objective function at :math:`c` and :math:`d`.

4. Update the interval:

   - If :math:`f(c) < f(d)`, set :math:`b = d`.

   - If :math:`f(c) \geq f(d)`, set :math:`a = c`.

5. Repeat steps 2-4 until the interval :math:`[a, b]` is sufficiently small.

Bisection Method
----------------
The Bisection Method is a simple yet effective line search technique that repeatedly divides the interval into two halves to locate the minimum.

Algorithm:

1. Choose an initial interval :math:`[a, b]` such that the minimum lies within this interval.

2. Compute the midpoint:

   .. math:: 

      m = \frac{a + b}{2}

3. Evaluate the objective function at :math:`m`.

4. Update the interval:

   - If :math:`f'(m) = 0`, :math:`m` is the minimum.

   - If :math:`f'(m) > 0`, set :math:`b = m`.

   - If :math:`f'(m) < 0`, set :math:`a = m`.

5. Repeat steps 2-4 until the interval :math:`[a, b]` is sufficiently small.

Armijo Rule
-----------

The Armijo Rule is a condition used to determine the step size in gradient descent methods. It ensures sufficient decrease in the objective function.

Algorithm:
1. Choose parameters :math:`\sigma \in (0, 1)` and :math:`\beta \in (0, 1)`.

2. Set the initial step size :math:`\alpha = 1`.

3. Check the Armijo condition:

   .. math:: 

      f(\mathbf{x}_k + \alpha \mathbf{d}_k) \leq f(\mathbf{x}_k) + \sigma \alpha \nabla f(\mathbf{x}_k)^T \mathbf{d}_k

4. If the condition is not satisfied, reduce the step size: :math:`alpha = \beta \alpha`.

5. Repeat step 3-4 until the condition is satisfied.

Wolfe Conditions
----------------
The Wolfe Conditions are a set of criteria used to ensure both sufficient decrease and curvature conditions for the step size.

Algorithm:
1. Choose parameters :math:`0 < \sigma_1 < \sigma_2 < 1`.

2. Set the initial step size :math:`\alpha = 1`.

3. Check the Wolfe conditions:

   - Sufficient decrease (Armijo condition):

     .. math:: 
        
        f(\mathbf{x}_k + \alpha \mathbf{d}_k) \leq f(\mathbf{x}_k) + \sigma_1 \alpha \nabla f(\mathbf{x}_k)^T \mathbf{d}_k

   - Curvature condition:

     .. math:: 

        \nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^T \mathbf{d}_k \geq \sigma_2 \nabla f(\mathbf{x}_k)^T \mathbf{d}_k

4. If the conditions are not satisfied, adjust the step size \(\alpha\).

5. Repeat step 3-4 until both conditions are satisfied.

These line search methods help ensure that the optimization process converges efficiently and effectively by selecting appropriate step sizes.