import time
import tracemalloc

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer


class CrossEntropy(BaseOptimizer):
    def optimize(self, bounds, pop_size=50, elite_frac=0.2, alpha=0.7, min_std=1e-6):
        """
        Perform Cross-Entropy Method optimization.

        Reference:
            Rubinstein, R. Y. (1999). "The Cross-Entropy Method for Combinatorial
            and Continuous Optimization." Methodology and Computing in Applied
            Probability, 1(2), 127-190.

        Args:
            bounds (list): List of (min, max) tuples for each dimension
            pop_size (int): Population size / number of samples per iteration (default: 50)
            elite_frac (float): Fraction of elite samples to use for update (default: 0.2)
            alpha (float): Smoothing parameter for mean/std update (default: 0.7)
            min_std (float): Minimum standard deviation to prevent premature convergence (default: 1e-6)

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

        # Initialize std as fraction of range
        std = (ub - lb) / 4.0

        # Number of elite samples
        n_elite = max(1, int(pop_size * elite_frac))

        # Random key for JAX
        key = jax.random.PRNGKey(42)

        # Track optimization path
        path = [mean.copy()]
        best_solution = mean.copy()
        best_fitness = self.penalized_obj(mean)

        # Progress tracking
        progress_bar = (
            tqdm(range(self.max_iter), desc="Cross-Entropy")
            if self.verbose
            else range(self.max_iter)
        )

        for k in progress_bar:
            # Sample population from Gaussian distribution
            key, subkey = jax.random.split(key)
            samples = mean + std * jax.random.normal(subkey, shape=(pop_size, n))

            # Clip to bounds
            samples = jnp.clip(samples, lb, ub)

            # Evaluate fitness
            fitness = jnp.array([self.penalized_obj(ind) for ind in samples])

            # Sort by fitness (ascending for minimization)
            sorted_indices = jnp.argsort(fitness)

            # Select elite samples
            elite_indices = sorted_indices[:n_elite]
            elite_samples = samples[elite_indices]

            # Update best solution
            if fitness[sorted_indices[0]] < best_fitness:
                best_fitness = fitness[sorted_indices[0]]
                best_solution = samples[sorted_indices[0]].copy()

            # Compute elite mean and std
            elite_mean = jnp.mean(elite_samples, axis=0)
            elite_std = jnp.std(elite_samples, axis=0)

            # Smooth update of mean and std
            mean = alpha * elite_mean + (1 - alpha) * mean
            std = alpha * elite_std + (1 - alpha) * std

            # Ensure minimum std to prevent premature convergence
            std = jnp.maximum(std, min_std)

            # Store path
            path.append(best_solution.copy())

            # Update progress bar
            if self.verbose and hasattr(progress_bar, "set_postfix"):
                progress_bar.set_postfix({"best": f"{best_fitness:.6f}", "std": f"{jnp.mean(std):.4f}"})

            # Check for convergence (std too small)
            if jnp.all(std <= min_std):
                break

        end_time = time.time()
        elapsed_time = end_time - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "method_name": "Cross-Entropy" if not self.f_cons else "Cross-Entropy with Penalty",
            "x0": (lb + ub) / 2.0,
            "xopt": best_solution,
            "fmin": best_fitness,
            "num_iter": k + 1,
            "path": jnp.array(path),
            "time": elapsed_time,
            "memory_peak": peak / 1e6,
        }


def cross_entropy(
    bounds=[(-5, 5), (-5, 5)],  # noqa
    pop_size=50,
    elite_frac=0.2,
    alpha=0.7,
    min_std=1e-6,
    **kwargs,
):
    """
    Cross-Entropy Method optimization algorithm.

    A population-based stochastic optimization method that iteratively updates
    a probability distribution to focus on promising regions of the search space.
    It samples from a Gaussian distribution, selects the best samples (elite),
    and updates the distribution parameters based on them.

    Reference:
        Rubinstein, R. Y. (1999). "The Cross-Entropy Method for Combinatorial
        and Continuous Optimization." Methodology and Computing in Applied
        Probability, 1(2), 127-190.

    Args:
        bounds (list): List of (min, max) tuples for each dimension
        pop_size (int): Population size / number of samples per iteration (default: 50)
        elite_frac (float): Fraction of elite samples to use for update (default: 0.2)
        alpha (float): Smoothing parameter for mean/std update (default: 0.7)
        min_std (float): Minimum standard deviation to prevent premature convergence (default: 1e-6)
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
    optimizer = CrossEntropy(**kwargs)
    result = optimizer.optimize(bounds, pop_size, elite_frac, alpha, min_std)
    return result
