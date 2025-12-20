import time
import tracemalloc

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer


class SimulatedAnnealing(BaseOptimizer):
    def optimize(self, bounds, T_init=1.0, T_min=1e-10, alpha=0.95, step_size=0.1):
        """
        Perform Simulated Annealing optimization.

        Args:
            bounds (list): List of (min, max) tuples for each dimension
            T_init (float): Initial temperature (default: 1.0)
            T_min (float): Minimum temperature / stopping criterion (default: 1e-10)
            alpha (float): Cooling rate, 0 < alpha < 1 (default: 0.95)
            step_size (float): Step size for generating neighbors (default: 0.1)

        Returns:
            dict: Optimization results
        """
        start_time = time.time()
        tracemalloc.start()

        # Problem dimension
        n = len(bounds)
        lb, ub = jnp.array(bounds).T

        # Initialize solution at center of bounds or use x0 if provided
        if self.x0 is not None:
            current = jnp.array(self.x0)
        else:
            current = (lb + ub) / 2.0

        current_energy = self.penalized_obj(current)

        # Best solution tracking
        best = current.copy()
        best_energy = current_energy

        # Temperature
        T = T_init

        # Random key for JAX
        key = jax.random.PRNGKey(42)

        # Track optimization path
        path = [current.copy()]

        # Progress tracking
        progress_bar = (
            tqdm(range(self.max_iter), desc="Simulated Annealing")
            if self.verbose
            else range(self.max_iter)
        )

        iteration = 0
        for k in progress_bar:
            iteration = k + 1

            # Check temperature stopping criterion
            if T < T_min:
                break

            # Generate neighbor solution
            key, subkey = jax.random.split(key)
            perturbation = jax.random.uniform(subkey, shape=(n,), minval=-1.0, maxval=1.0)
            neighbor = current + step_size * (ub - lb) * perturbation

            # Clip to bounds
            neighbor = jnp.clip(neighbor, lb, ub)

            # Evaluate neighbor
            neighbor_energy = self.penalized_obj(neighbor)

            # Compute energy difference
            delta_energy = neighbor_energy - current_energy

            # Accept or reject
            key, subkey = jax.random.split(key)
            random_val = jax.random.uniform(subkey)

            # Accept if better, or with probability exp(-delta/T) if worse
            accept = (delta_energy < 0) | (random_val < jnp.exp(-delta_energy / T))

            if accept:
                current = neighbor
                current_energy = neighbor_energy

                # Update best if improved
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy

            # Cool down
            T = T * alpha

            # Store path
            path.append(best.copy())

            # Update progress bar
            if self.verbose and hasattr(progress_bar, "set_postfix"):
                progress_bar.set_postfix({"T": f"{T:.2e}", "best": f"{best_energy:.6f}"})

        end_time = time.time()
        elapsed_time = end_time - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "method_name": "Simulated Annealing" if not self.f_cons else "Simulated Annealing with Penalty",
            "x0": (lb + ub) / 2.0 if self.x0 is None else jnp.array(self.x0),
            "xopt": best,
            "fmin": best_energy,
            "num_iter": iteration,
            "path": jnp.array(path),
            "time": elapsed_time,
            "memory_peak": peak / 1e6,
        }


def simulated_annealing(
    bounds=[(-5, 5), (-5, 5)],  # noqa
    T_init=1.0,
    T_min=1e-10,
    alpha=0.95,
    step_size=0.1,
    **kwargs,
):
    """
    Simulated Annealing optimization algorithm.

    A probabilistic optimization technique inspired by the annealing process in metallurgy.
    It explores the search space by accepting worse solutions with a probability that
    decreases as the temperature cools down, allowing escape from local minima.

    Args:
        bounds (list): List of (min, max) tuples for each dimension
        T_init (float): Initial temperature (default: 1.0)
        T_min (float): Minimum temperature / stopping criterion (default: 1e-10)
        alpha (float): Cooling rate, 0 < alpha < 1 (default: 0.95)
        step_size (float): Step size for generating neighbors as fraction of range (default: 0.1)
        **kwargs: Additional arguments passed to BaseOptimizer (f_obj, f_cons, max_iter, verbose, etc.)

    Returns:
        dict: Optimization results containing:
            - method_name: Name of the method
            - x0: Initial point
            - xopt: Optimal solution found
            - fmin: Minimum function value
            - num_iter: Number of iterations
            - path: Optimization path
            - time: Elapsed time
            - memory_peak: Peak memory usage in MB
    """
    optimizer = SimulatedAnnealing(**kwargs)
    result = optimizer.optimize(bounds, T_init, T_min, alpha, step_size)
    return result
