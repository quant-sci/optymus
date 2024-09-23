import warnings

import jax.numpy as jnp

from optymus.plots import plot_function

warnings.filterwarnings('ignore')


class MccormickFunction:
    r"""McCormick Function

    The McCormick function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = sin(x_0 + x_1) + (x_0 - x_1)^2 - 1.5x_0 + 2.5x_1 + 1

    Reference:
        https://www.sfu.ca/~ssurjano/mccorm.html
    """
    def __init__(self):
        self.name = 'McCormick Function'

    def min_point(self):
        return [-0.54719, -1.54719]

    def __call__(self, x):
        x = jnp.array(x)
        return jnp.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

    def plot(self):
        plot_function(self, min=-5.12, max=5.12)

class MccormickFunction:
    r"""McCormick Function

    The McCormick function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = sin(x_0 + x_1) + (x_0 - x_1)^2 - 1.5x_0 + 2.5x_1 + 1

    Reference:
    https://www.sfu.ca/~ssurjano/mccorm.html
    """
    def __init__(self):
        self.name = 'McCormick Function'

    def min_point(self):
        return [-0.54719, -1.54719]

    def __call__(self, x):
        x = jnp.array(x)
        return jnp.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

    def plot(self):
        plot_function(self, min=-5.12, max=5.12)


class RastriginFunction:
    r"""Rastrigin Function

    The Rastrigin function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = 20 + x_0^2 + x_1^2 - 10(\cos(2\pi x_0) + \cos(2\pi x_1))

    Reference:
    https://www.sfu.ca/~ssurjano/rastr.html
    """
    def __init__(self):
        self.name = 'Rastrigin Function'

    def __call__(self, x):
        x = jnp.array(x)
        return 20 + x[0]**2 + x[1]**2 - 10*(jnp.cos(2*jnp.pi*x[0]) + jnp.cos(2*jnp.pi*x[1]))

    def plot(self):
        plot_function(self, min=-5.12, max=5.12)

class AckleyFunction:
    r"""Ackley Function

    The Ackley function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = -20\exp(-0.2\sqrt{0.5(x_0^2 + x_1^2)}) - \exp(0.5(\cos(2\pi x_0) + \cos(2\pi x_1))) + e + 20

    Reference:
        https://www.sfu.ca/~ssurjano/ackley.html
    """
    def __init__(self):
        self.name = 'Ackley Function'

    def __call__(self, x):
        x = jnp.array(x)
        return -20*jnp.exp(-0.2*jnp.sqrt(0.5*(x[0]**2 + x[1]**2))) - jnp.exp(0.5*(jnp.cos(2*jnp.pi*x[0]) + jnp.cos(2*jnp.pi*x[1]))) + jnp.exp(1) + 20  # noqa: E501

    def plot(self):
        plot_function(self, min=-5.12, max=5.12)

class EggholderFunction:
    r"""Eggholder Function

    The Eggholder function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = -(x_1 + 47)\sin(\sqrt{|x_0/2 + x_1 + 47|}) - x_0\sin(\sqrt{|x_0 - x_1 - 47|})

    Reference:
    https://www.sfu.ca/~ssurjano/egg.html

    """
    def __init__(self):
        self.name = 'Eggholder Function'

    def __call__(self, x):
        x = jnp.array(x)
        return -(x[1] + 47)*jnp.sin(jnp.sqrt(jnp.abs(x[0]/2 + x[1] + 47))) \
            - x[0]*jnp.sin(jnp.sqrt(jnp.abs(x[0] - x[1] - 47)))

    def plot(self):
        plot_function(self, min=-512, max=512)

class CrossintrayFunction:
    r"""Crossintray Function

    The Crossintray function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = -0.0001(|\sin(x_0)\sin(x_1)\exp(|100 - \sqrt{x_0^2 + x_1^2}/\pi|) + 1)^{0.1}

    Reference:
    https://www.sfu.ca/~ssurjano/crossit.html

    """
    def __init__(self):
        self.name = 'Crossintray Function'

    def __call__(self, x):
        x = jnp.array(x)
        return -0.0001 * (
            jnp.abs(jnp.sin(x[0]) * jnp.sin(x[1]) * jnp.exp(jnp.abs(100 - jnp.sqrt(x[0]**2 + x[1]**2) / jnp.pi))) + 1
        ) ** 0.1

    def plot(self):
        plot_function(self, min=-10, max=10)

class SphereFunction:
    r"""Sphere Function

    The Sphere function is a convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = x_0^2 + x_1^2

    Reference:
    https://www.sfu.ca/~ssurjano/spheref.html
    """
    def __init__(self):
        self.name = 'Sphere Function'

    def __call__(self, x):
        x = jnp.array(x)
        return x[0]**2+x[1]**2

    def plot(self):
        plot_function(self, min=-5.12, max=5.12)

class RosenbrockFunction:
    r"""Rosenbrock Function

    The Rosenbrock function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Reference:
    https://www.sfu.ca/~ssurjano/rosen.html
    """
    def __init__(self):
        self.name = 'Rosenbrock Function'

    def __call__(self, x):
        x = jnp.array(x)
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    def plot(self):
        plot_function(self, min=-5.12, max=5.12)

class BealeFunction:
    r"""Beale Function

    The Beale function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = (1.5 - x_0 + x_0x_1)^2 + (2.25 - x_0 + x_0x_1^2)^2 + (2.625 - x_0 + x_0x_1^3)^2

    Reference:
        https://www.sfu.ca/~ssurjano/beale.html
    """
    def __init__(self):
        self.name = 'Beale Function'

    def __call__(self, x):
        x = jnp.array(x)
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

    def plot(self):
        plot_function(self, min=-4.5, max=4.5)

class SemionescuFunction:
    r"""Semionescu Function

    The Semionescu function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = 0.1(x_0^2 + x_1^2) - 0.3cos(3\pi x_0)cos(4\pi x_1) + 0.3

    Reference:
        https://www.sfu.ca/~ssurjano/semion.html
    """
    def __init__(self):
        self.name = 'Semionescu Function'

    def __call__(self, x):
        x = jnp.array(x)
        return 0.1*(x[0]**2 + x[1]**2) - 0.3*jnp.cos(3*jnp.pi*x[0])*jnp.cos(4*jnp.pi*x[1]) + 0.3

    def plot(self):
        plot_function(self, min=-5, max=5)

class GoldsteinPriceFunction:
    r"""Goldstein Price Function

    The Goldstein-Price function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = [1 + (x_0 + x_1 + 1)^2(19 - 14x_0 + 3x_0^2 - 14x_1 + 6x_0x_1 + 3x_1^2)]
                [30 + (2x_0 - 3x_1)^2(18 - 32x_0 + 12x_0^2 + 48x_1 - 36x_0x_1 + 27x_1^2)]

    Reference:
        https://www.sfu.ca/~ssurjano/goldpr.html
    """
    def __init__(self):
        self.name = 'Goldenstein Price Function'

    def __call__(self, x):
        x = jnp.array(x)
        return (
            (1 + (x[0] + x[1] + 1)**2*(19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) *
            (30 + (2*x[0] - 3*x[1])**2*(18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
        )

    def plot(self):
        plot_function(self, min=-2, max=2)

class BoothFunction:
    r"""Booth Function

    The Booth function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = (x_0 + 2x_1 - 7)^2 + (2x_0 + x_1 - 5)^2

    Reference:
        https://www.sfu.ca/~ssurjano/booth.html
    """
    def __init__(self):
        self.name = 'Booth Function'

    def __call__(self, x):
        x = jnp.array(x)
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def plot(self):
        plot_function(self, min=-10, max=10)

class StyblinskiTangFunction:
    r"""Styblinski-Tang Function

    The Styblinski-Tang function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = 0.5((x_0^4 - 16x_0^2 + 5x_0) + (x_1^4 - 16x_1^2 + 5x_1))

    Reference:
        https://www.sfu.ca/~ssurjano/stybtang.html
    """
    def __init__(self):
        self.name = 'Styblinski-Tang'

    def __call__(self, x):
        x = jnp.array(x)
        return 0.5*((x[0]**4 - 16*x[0]**2 + 5*x[0])+(x[1]**4 - 16*x[1]**2 + 5*x[1]))

    def plot(self):
        plot_function(self, min=-5, max=5)

class HimmeblauFunction:
    r"""Himmeblau Function

    The Himmeblau function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    .. math::
        f(x) = (x_0^2 + x_1 - 11)^2 + (x_0 + x_1^2 - 7)^2
    """
    def __init__(self):
        self.name = 'Himmeblau Function'

    def __call__(self, x):
        x = jnp.array(x)
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def plot(self, **kwargs):
        plot_function(self, **kwargs)

class CustomFunction:
    """Custom Function

    Parameters
    ----------

    f : callable
        A callable function that takes a single argument.
    """
    def __init__(self, f):
        self.f = f
        self.name = 'Custom Function'

    def __call__(self, x):
        return self.f(x)

    def plot(self):
        plot_function(self, min=-5.12, max=5.12)
