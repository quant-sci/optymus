import time
import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from optymus.methods.utils import BaseOptimizer
from optymus.methods.utils._result import OptimizeResult


class DifferentialEvolution(BaseOptimizer):
    def optimize(self, bounds, mutation_factor=0.8, crossover_prob=0.7, pop_size=50):
        """
        Perform Differential Evolution optimization

        Args:
            bounds (list): Lower and upper bounds for each dimension
            mutation_factor (float): Mutation scaling factor
            crossover_prob (float): Crossover probability
            pop_size (int): Number of individuals in X

        Returns:
            dict: Optimization results
        """
        start_time = time.time()
        tracemalloc.start()

        # Ensure bounds are jax numpy arrays
        dimensions = len(bounds)
        lb, ub = jnp.array(bounds).T

        # Set random seed for reproducibility
        key = jax.random.PRNGKey(42)

        # Initialize X randomly within bounds
        X = jax.random.uniform(key, shape=(pop_size, dimensions), minval=lb, maxval=ub)

        # Initial fitness evaluation
        fitness_values = jnp.array([self.penalized_obj(individual) for individual in X])

        # Track optimization path
        path = [[] for _ in range(self.max_iter)]
        path_gbest = []

        # Find initial global best
        best_idx = jnp.argmin(fitness_values)
        gbest = X[best_idx].copy()
        gbest_val = fitness_values[best_idx]
        f_history = [float(gbest_val)]

        # Progress tracking
        progress_bar = tqdm(range(self.max_iter), desc="Differential Evolution", disable=not self.verbose)

        for k in progress_bar:
            # Iterate over the population
            for i in range(pop_size):
                x = X[i]

                # Sampling without replacement, excluding x
                candidates = jnp.array([j for j in range(pop_size) if j != i])
                key, subkey = jax.random.split(key)
                abc_indices = jax.random.choice(subkey, candidates, shape=(3,), replace=False)

                a, b, c = X[abc_indices[0]], X[abc_indices[1]], X[abc_indices[2]]
                z = a + mutation_factor * (b - c)

                # Generate random numbers for crossover probability
                key, subkey = jax.random.split(key)
                j = jax.random.randint(subkey, shape=(), minval=0, maxval=dimensions)
                rand_vals = jax.random.uniform(subkey, shape=(dimensions,))

                # Apply crossover logic
                mask = (jnp.arange(dimensions) == j) | (rand_vals < crossover_prob)
                x_prime = jnp.where(mask, z, x)

                # Evaluate fitness and update individual
                x_prime_fitness = self.penalized_obj(x_prime)
                if x_prime_fitness < fitness_values[i]:
                    X = X.at[i].set(x_prime)
                    fitness_values = fitness_values.at[i].set(x_prime_fitness)

                # Update global best
                gbest_idx = jnp.argmin(fitness_values)
                if fitness_values[gbest_idx] < gbest_val:
                    gbest = X[gbest_idx].copy()
                    gbest_val = fitness_values[gbest_idx]

            # Store X state and global best
            path[k].append(X.copy())
            path_gbest.append(gbest.copy())
            f_history.append(float(gbest_val))

        end_time = time.time()
        elapsed_time = end_time - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return OptimizeResult({
            "method_name": "Differential Evolution" if not self.f_cons else "Differential Evolution with Penalty",
            "x0": self.x0,
            "xopt": gbest,
            "fmin": gbest_val,
            "num_iter": self.max_iter,
            "path_particles": path,
            "path": jnp.array(path_gbest),
            "f_history": jnp.array(f_history),
            "termination_reason": "max_iter_reached",
            "time": elapsed_time,
            "memory_peak": peak / 1e6,
        })



def differential_evolution(
    bounds=[(-5, 5), (-5, 5)],  # noqa
    mutation_factor=0.2,
    crossover_prob=0.5,
    pop_size=30,
    **kwargs,
):
    """Particle Swarm Optimization algorithm."""
    optimizer = DifferentialEvolution(**kwargs)
    result = optimizer.optimize(bounds, mutation_factor, crossover_prob, pop_size)

    return result
