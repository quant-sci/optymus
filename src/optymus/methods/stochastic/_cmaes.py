import time
import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer


class CovarianceMatrixAdaptation(BaseOptimizer):
    def optimize(self, bounds, pop_size=None, sigma=0.5):
        """
        Perform CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization.

        Args:
            bounds (list): Lower and upper bounds for each dimension
            pop_size (int): Population size (lambda). If None, uses 4 + floor(3 * ln(n))
            sigma (float): Initial step-size

        Returns:
            dict: Optimization results
        """
        start_time = time.time()
        tracemalloc.start()

        # Problem dimension
        n = len(bounds)
        lb, ub = jnp.array(bounds).T

        # Initialize mean at center of bounds
        mean = (lb + ub) / 2.0

        # Population size (lambda)
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(n))

        # Number of parents for recombination (mu)
        mu = pop_size // 2

        # Recombination weights
        weights = jnp.log(mu + 0.5) - jnp.log(jnp.arange(1, mu + 1))
        weights = weights / jnp.sum(weights)
        mu_eff = 1.0 / jnp.sum(weights**2)

        # Learning rates for covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, jnp.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Expected value of ||N(0,I)||
        chi_n = jnp.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # Initialize evolution paths
        p_c = jnp.zeros(n)
        p_sigma = jnp.zeros(n)

        # Initialize covariance matrix
        C = jnp.eye(n)
        B = jnp.eye(n)
        D = jnp.ones(n)

        # Random key for JAX
        key = jax.random.PRNGKey(42)

        # Track optimization path
        path = []
        best_solution = mean.copy()
        best_fitness = self.penalized_obj(mean)

        # Progress tracking
        progress_bar = (
            tqdm(range(self.max_iter), desc="CMA-ES") if self.verbose else range(self.max_iter)
        )

        for k in progress_bar:
            # Sample population
            key, subkey = jax.random.split(key)
            z = jax.random.normal(subkey, shape=(pop_size, n))
            y = z @ (B * D).T
            x = mean + sigma * y

            # Clip to bounds
            x = jnp.clip(x, lb, ub)

            # Evaluate fitness
            fitness = jnp.array([self.penalized_obj(ind) for ind in x])

            # Sort by fitness (ascending for minimization)
            sorted_indices = jnp.argsort(fitness)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            z_sorted = z[sorted_indices]

            # Update best solution
            if fitness[sorted_indices[0]] < best_fitness:
                best_fitness = fitness[sorted_indices[0]]
                best_solution = x_sorted[0].copy()

            # Select mu best individuals
            x_mu = x_sorted[:mu]
            y_mu = y_sorted[:mu]
            z_mu = z_sorted[:mu]

            # Update mean
            mean_old = mean
            mean = jnp.sum(weights[:, None] * x_mu, axis=0)

            # Update evolution path for sigma
            y_mean = jnp.sum(weights[:, None] * y_mu, axis=0)
            z_mean = jnp.sum(weights[:, None] * z_mu, axis=0)

            p_sigma = (1 - c_sigma) * p_sigma + jnp.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ z_mean)

            # Heaviside function for stalling
            h_sigma = (
                jnp.linalg.norm(p_sigma) / jnp.sqrt(1 - (1 - c_sigma) ** (2 * (k + 1)))
                < (1.4 + 2 / (n + 1)) * chi_n
            )

            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + h_sigma * jnp.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Adapt covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = jnp.outer(p_c, p_c)
            rank_mu = jnp.sum(
                jnp.array([weights[i] * jnp.outer(y_mu[i], y_mu[i]) for i in range(mu)]),
                axis=0,
            )
            C = (1 - c_1 - c_mu + delta_h * c_1) * C + c_1 * rank_one + c_mu * rank_mu

            # Adapt step-size
            sigma = sigma * jnp.exp((c_sigma / d_sigma) * (jnp.linalg.norm(p_sigma) / chi_n - 1))

            # Eigendecomposition of C for next iteration
            C = (C + C.T) / 2  # Ensure symmetry
            eigenvalues, B = jnp.linalg.eigh(C)
            eigenvalues = jnp.maximum(eigenvalues, 1e-20)  # Numerical stability
            D = jnp.sqrt(eigenvalues)

            # Store path
            path.append(best_solution.copy())

        end_time = time.time()
        elapsed_time = end_time - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "method_name": "CMA-ES" if not self.f_cons else "CMA-ES with Penalty",
            "x0": (lb + ub) / 2.0,
            "xopt": best_solution,
            "fmin": best_fitness,
            "num_iter": self.max_iter,
            "path": jnp.array(path),
            "time": elapsed_time,
            "memory_peak": peak / 1e6,
        }


def cmaes(
    bounds=[(-5, 5), (-5, 5)],  # noqa
    pop_size=None,
    sigma=0.5,
    **kwargs,
):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm.

    A state-of-the-art evolutionary strategy for difficult non-linear, non-convex
    optimization problems. It adapts the covariance matrix of a multivariate normal
    distribution to learn the structure of the search space.

    Args:
        bounds (list): List of (min, max) tuples for each dimension
        pop_size (int): Population size. If None, uses 4 + floor(3 * ln(n))
        sigma (float): Initial step-size (default: 0.5)
        **kwargs: Additional arguments passed to BaseOptimizer (f_obj, f_cons, max_iter, verbose, etc.)

    Returns:
        dict: Optimization results containing:
            - method_name: Name of the method
            - x0: Initial point (center of bounds)
            - xopt: Optimal solution found
            - fmin: Minimum function value
            - num_iter: Number of iterations
            - path: Optimization path
            - time: Elapsed time
            - memory_peak: Peak memory usage in MB
    """
    optimizer = CovarianceMatrixAdaptation(**kwargs)
    result = optimizer.optimize(bounds, pop_size, sigma)
    return result
