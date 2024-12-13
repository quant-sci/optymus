import warnings

import jax.numpy as jnp

from optymus.plots import plot_function

warnings.filterwarnings("ignore")


class WheelersRidge:
    r"""Wheeler's Ridge Function

    The Wheeler's Ridge function is two-dimensional function with a single global minimum in a deep curved peak.

    It is defined as:

    .. math::
        f(x) = -\exp (-(x_{1}x_{2}-a)^{2}-(x_{2}-a)^{2})

    where :math:`a` is typically set to 1.5 and the global minimum is at :math:`x = [1, 3/2]`.
    """

    NAME = "Wheeler's Ridge Function"
    TRUE_MINIMUM = [1, 1.5]
    BOUNDS = [(0, 3), (0, 3)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        a = 1.5
        return -jnp.exp(-((x[0] * x[1] - a) ** 2) - (x[1] - a) ** 2)

    def plot(self):
        plot_function(self, lb=-10, ub=10)


class Levy:
    r"""Levy Function

    The Levy function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = \sin^2(\pi w_1) + \sum_{i=1}^{d-1} (w_i - 1)^2 [1 + 10\sin^2(\pi w_i + 1)] + (w_d - 1)^2 [1 + \sin^2(2\pi w_d)]

    where :math:`w_i = 1 + \frac{x_i - 1}{4}`

    Reference:
        https://www.sfu.ca/~ssurjano/levy.html
    """

    NAME = "Levy Function"
    TRUE_MINIMUM = [1, 1]
    BOUNDS = [(-10, 10), (-10, 10)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        w = 1 + (x - 1) / 4
        return (
            jnp.sin(jnp.pi * w[0]) ** 2
            + jnp.sum((w[:-1] - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * w[:-1] + 1) ** 2))
            + (w[-1] - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * w[-1]) ** 2)
        )

    def plot(self):
        plot_function(self, lb=-10, ub=10)


class Mccormick:
    r"""McCormick Function

    The McCormick function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = sin(x_0 + x_1) + (x_0 - x_1)^2 - 1.5x_0 + 2.5x_1 + 1

    Reference:
        https://www.sfu.ca/~ssurjano/mccorm.html
    """

    NAME = "McCormick Function"
    TRUE_MINIMUM = [-0.54719, -1.54719]
    BOUNDS = [(-1.5, 4), (-3, 4)]

    def __init__(self):
        pass

    def min_point(self):
        return [-0.54719, -1.54719]

    def __call__(self, x):
        x = jnp.array(x)
        return jnp.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1

    def plot(self):
        plot_function(self, lb=-5.12, ub=5.12)


class Rastrigin:
    r"""Rastrigin Function

    The Rastrigin function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = 20 + x_0^2 + x_1^2 - 10(\cos(2\pi x_0) + \cos(2\pi x_1))

    Reference:
    https://www.sfu.ca/~ssurjano/rastr.html
    """

    NAME = "Rastrigin Function"
    TRUE_MINIMUM = [0, 0]
    BOUNDS = [(-5.12, 5.12), (-5.12, 5.12)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return 20 + x[0] ** 2 + x[1] ** 2 - 10 * (jnp.cos(2 * jnp.pi * x[0]) + jnp.cos(2 * jnp.pi * x[1]))

    def plot(self):
        plot_function(self, lb=-5.12, ub=5.12)


class Ackley:
    r"""Ackley Function

    The Ackley function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = -20\exp(-0.2\sqrt{0.5(x_0^2 + x_1^2)}) - \exp(0.5(\cos(2\pi x_0) + \cos(2\pi x_1))) + e + 20

    Reference:
        https://www.sfu.ca/~ssurjano/ackley.html
    """

    NAME = "Ackley Function"
    TRUE_MINIMUM = [0, 0]
    BOUNDS = [(-5, 5), (-5, 5)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return (
            -20 * jnp.exp(-0.2 * jnp.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
            - jnp.exp(0.5 * (jnp.cos(2 * jnp.pi * x[0]) + jnp.cos(2 * jnp.pi * x[1])))
            + jnp.exp(1)
            + 20
        )

    def plot(self):
        plot_function(self, lb=-5.12, ub=5.12)


class Eggholder:
    r"""Eggholder Function

    The Eggholder function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = -(x_1 + 47)\sin(\sqrt{|x_0/2 + x_1 + 47|}) - x_0\sin(\sqrt{|x_0 - x_1 - 47|})

    Reference:
    https://www.sfu.ca/~ssurjano/egg.html

    """

    NAME = "Eggholder Function"
    TRUE_MINIMUM = [512, 404.2319]
    BOUNDS = [(-512, 512), (-512, 512)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return -(x[1] + 47) * jnp.sin(jnp.sqrt(jnp.abs(x[0] / 2 + x[1] + 47))) - x[0] * jnp.sin(
            jnp.sqrt(jnp.abs(x[0] - x[1] - 47))
        )

    def plot(self):
        plot_function(self, lb=-512, ub=512)


class Crossintray:
    r"""Crossintray Function

    The Crossintray function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = -0.0001(|\sin(x_0)\sin(x_1)\exp(|100 - \sqrt{x_0^2 + x_1^2}/\pi|) + 1)^{0.1}

    Reference:
    https://www.sfu.ca/~ssurjano/crossit.html
    """

    NAME = "Crossintray Function"
    TRUE_MINIMUM = [1.34941, 1.34941]
    BOUNDS = [(-10, 10), (-10, 10)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return (
            -0.0001
            * (
                jnp.abs(
                    jnp.sin(x[0]) * jnp.sin(x[1]) * jnp.exp(jnp.abs(100 - jnp.sqrt(x[0] ** 2 + x[1] ** 2) / jnp.pi))
                )
                + 1
            )
            ** 0.1
        )

    def plot(self):
        plot_function(self, lb=-10, ub=10)


class Sphere:
    r"""Sphere Function

    The Sphere function is a convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = x_0^2 + x_1^2

    Reference:
    https://www.sfu.ca/~ssurjano/spheref.html
    """

    NAME = "Sphere Function"
    TRUE_MINIMUM = [0, 0]
    BOUNDS = [(-5.12, 5.12), (-5.12, 5.12)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return x[0] ** 2 + x[1] ** 2

    def plot(self):
        plot_function(self, lb=-5.12, ub=5.12)


class Rosenbrock:
    r"""Rosenbrock Function

    The Rosenbrock function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Reference:
    https://www.sfu.ca/~ssurjano/rosen.html
    """

    NAME = "Rosenbrock Function"
    TRUE_MINIMUM = [1, 1]
    BOUNDS = [(-5, 10)]

    def __init__(self, dimension=2):
        """
        Initialize the Rosenbrock function.

        Args:
            dimension (int): The number of dimensions (variables) of the function.
        """
        self.dimension = dimension
        self.TRUE_MINIMUM = [1] * dimension
        self.BOUNDS = [(-5, 10)] * dimension

    def __call__(self, x):
        """
        Evaluate the Rosenbrock function at a given point.

        Args:
            x (array-like): Input array of shape (dimension,).

        Returns:
            float: The function value at x.
        """
        x = jnp.array(x)
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def plot(self):
        """
        Plot a 2D slice of the Rosenbrock function.
        For higher dimensions, the first two variables are varied, keeping others fixed at their true minimum.
        """
        if self.dimension > 2:
            print("Plotting a 2D slice with other dimensions fixed at their true minimum.")  # noqa

        plot_function(self, lb=-5.12, ub=5.12)


class Beale:
    r"""Beale Function

    The Beale function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = (1.5 - x_0 + x_0x_1)^2 + (2.25 - x_0 + x_0x_1^2)^2 + (2.625 - x_0 + x_0x_1^3)^2

    Reference:
        https://www.sfu.ca/~ssurjano/beale.html
    """

    NAME = "Beale Function"
    TRUE_MINIMUM = [3, 0.5]
    BOUNDS = [(-4.5, 4.5), (-4.5, 4.5)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return (
            (1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        )

    def plot(self):
        plot_function(self, lb=-4.5, ub=4.5)


class Semionescu:
    r"""Semionescu Function

    The Semionescu function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = 0.1(x_0^2 + x_1^2) - 0.3cos(3\pi x_0)cos(4\pi x_1) + 0.3

    Reference:
        https://www.sfu.ca/~ssurjano/semion.html
    """

    NAME = "Semionescu Function"
    TRUE_MINIMUM = [0.84852813, 0.84852813]
    BOUNDS = [(-1.25, 1.25), (-1.25, 1.25)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return 0.1 * (x[0] ** 2 + x[1] ** 2) - 0.3 * jnp.cos(3 * jnp.pi * x[0]) * jnp.cos(4 * jnp.pi * x[1]) + 0.3

    def plot(self):
        plot_function(self, lb=-5, ub=5)


class GoldsteinPrice:
    r"""Goldstein Price Function

    The Goldstein-Price function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = [1 + (x_0 + x_1 + 1)^2(19 - 14x_0 + 3x_0^2 - 14x_1 + 6x_0x_1 + 3x_1^2)]
                [30 + (2x_0 - 3x_1)^2(18 - 32x_0 + 12x_0^2 + 48x_1 - 36x_0x_1 + 27x_1^2)]

    Reference:
        https://www.sfu.ca/~ssurjano/goldpr.html
    """

    NAME = "Goldstein-Price Function"
    TRUE_MINIMUM = [0, -1]
    BOUNDS = [(-2, 2), (-2, 2)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return (
            1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
        ) * (
            30
            + (2 * x[0] - 3 * x[1]) ** 2
            * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
        )

    def plot(self):
        plot_function(self, lb=-2, ub=2)


class Booth:
    r"""Booth Function

    The Booth function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = (x_0 + 2x_1 - 7)^2 + (2x_0 + x_1 - 5)^2

    Reference:
        https://www.sfu.ca/~ssurjano/booth.html
    """

    NAME = "Booth Function"
    TRUE_MINIMUM = [1, 3]
    BOUNDS = [(-10, 10), (-10, 10)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

    def plot(self):
        plot_function(self, lb=-10, ub=10)


class StyblinskiTang:
    r"""Styblinski-Tang Function

    The Styblinski-Tang function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = 0.5((x_0^4 - 16x_0^2 + 5x_0) + (x_1^4 - 16x_1^2 + 5x_1))

    Reference:
        https://www.sfu.ca/~ssurjano/stybtang.html
    """

    NAME = "Styblinski-Tang Function"
    TRUE_MINIMUM = [-2.903534, -2.903534]
    BOUNDS = [(-5, 5), (-5, 5)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return 0.5 * ((x[0] ** 4 - 16 * x[0] ** 2 + 5 * x[0]) + (x[1] ** 4 - 16 * x[1] ** 2 + 5 * x[1]))

    def plot(self):
        plot_function(self, lb=-5, ub=5)


class Himmeblau:
    r"""Himmeblau Function

    The Himmeblau function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = (x_0^2 + x_1 - 11)^2 + (x_0 + x_1^2 - 7)^2
    """

    NAME = "Himmeblau Function"
    TRUE_MINIMUM = [3, 2]
    BOUNDS = [(-6, 6), (-6, 6)]

    def __init__(self):
        pass

    def __call__(self, x):
        x = jnp.array(x)
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def plot(self, **kwargs):
        plot_function(self, lb=-6, ub=6)


class CustomFunction:
    """Custom Function

    Parameters
    ----------

    f : callable
        A callable function that takes a single argument.
    """

    NAME = "Custom Function"

    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def plot(self):
        plot_function(self, lb=-5.12, ub=5.12)
