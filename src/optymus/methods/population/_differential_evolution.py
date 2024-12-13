import time
import tracemalloc

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer


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

        # Progress tracking
        progress_bar = (
            tqdm(range(self.max_iter), desc="Differential Evolution") if self.verbose else range(self.max_iter)
        )

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

        end_time = time.time()
        elapsed_time = end_time - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "method_name": "Differential Evolution" if not self.f_cons else "Differential Evolution with Penalty",
            "x0": self.x0,
            "xopt": gbest,
            "fmin": gbest_val,
            "num_iter": self.max_iter,
            "path_particles": path,
            "path": jnp.array(path_gbest),
            "time": elapsed_time,
            "memory_peak": peak / 1e6,
        }


def visualize_diffevo(particle_paths, gbest_path, bounds, obj):
    """Visualizes PSO in 2D."""

    fig, ax = plt.subplots(figsize=(6, 5))
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = obj([X, Y])  # Evaluate the objective function on the grid
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cm.PuBuGn_r, alpha=0.8)
    fig.colorbar(contour)

    points = ax.scatter([], [], s=40, c="midnightblue", label="Particles")
    gbest_point = ax.scatter([], [], s=30, c="indianred", marker="*", label="Global Best")
    ax.legend(loc="lower left")

    iteration_text = plt.text(0.02, 0.95, "", transform=ax.transAxes)
    best_value_text = plt.text(0.02, 0.90, "", transform=ax.transAxes)

    def animate(frame):
        points.set_offsets(particle_paths[frame][0])
        gbest_point.set_offsets(gbest_path[frame].reshape(1, -1))
        iteration_text.set_text(f"Iteration: {frame+1}")

        best_value_text.set_text(f"Best Point: {gbest_path[frame]}")

        return points, gbest_point

    ani = FuncAnimation(fig, animate, frames=len(particle_paths), interval=50, blit=True, repeat=False)
    ani.save("diffevo_animation.gif", writer="Pillow", fps=2, dpi=300)  # Use pillow, imagemagick can be problematic

    plt.title("Differential Evolution Optimization")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.close()


def differential_evolution(
    bounds=[(-5, 5), (-5, 5)],  # noqa
    mutation_factor=0.2,
    crossover_prob=0.5,
    pop_size=30,
    visualize=False,
    **kwargs,
):
    """Particle Swarm Optimization algorithm."""
    optimizer = DifferentialEvolution(**kwargs)
    result = optimizer.optimize(bounds, mutation_factor, crossover_prob, pop_size)

    if visualize and len(bounds[0]) == 2:  # Only visualize if 2D
        visualize_diffevo(result["path_particles"], result["path"], bounds, optimizer.f_obj)

    return result
