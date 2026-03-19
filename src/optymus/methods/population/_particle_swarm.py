import time
import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import track

from optymus.methods.utils import BaseOptimizer


class ParticleSwarmOptimization(BaseOptimizer):
    def optimize(self, divergence, bounds, alpha, w, c1, c2, n_particles):
        start_time = time.time()
        tracemalloc.start()  # Start memory tracking

        num_iter = 0
        dimensions = len(bounds)
        lb, ub = jnp.array(bounds).T
        if n_particles is None:
            n_particles = max(30, int(10 * np.sqrt(dimensions)))

        # Initialize random number generator keys
        key = jax.random.PRNGKey(42)

        # Initialize particles' positions and velocities randomly within bounds
        def generate_particles(key):
            key_pos, key_vel = jax.random.split(key)
            X = jax.random.uniform(key_pos, shape=(n_particles, dimensions), minval=lb, maxval=ub)
            # More reasonable velocity initialization
            V = jax.random.uniform(
                key_vel, shape=(n_particles, dimensions), minval=-jnp.abs(ub - lb), maxval=jnp.abs(ub - lb)
            )
            return X, V

        X, V = generate_particles(key)

        path = [[] for _ in range(self.max_iter)]
        path_gbest = []
        particle_diversity = []
        velocity_diversity = []

        # Initialize particle best and global best
        pbest = X.copy()
        pbest_val = jnp.array([self.penalized_obj(x) for x in X])
        gbest = pbest[jnp.argmin(pbest_val)].copy()
        gbest_val = jnp.min(pbest_val)

        progres_bar = (
            track(range(self.max_iter), description="Particle Swarm Optimization") if self.verbose else range(self.max_iter)
        )

        for i in progres_bar:
            # Generate new keys for each iteration
            key, key_r1, key_r2 = jax.random.split(key, 3)

            # Generate r1 and r2
            r1 = jax.random.uniform(key_r1, shape=(n_particles, dimensions))
            r2 = jax.random.uniform(key_r2, shape=(n_particles, dimensions))

            # Compute divergence term
            if divergence == "baseline":
                divergence_term = jnp.zeros_like(X)
            else:
                # Normalize particle positions and global best to a probability distribution
                X_prob = jax.nn.softmax(pbest, axis=1)  # Softmax over dimensions for each particle
                gbest_prob = jax.nn.softmax(gbest)  # Softmax for the global best
                div = divergence(X_prob, gbest_prob)

                divergence_term = alpha * div[:, None] * jnp.sign(X - gbest)

            # Update velocities and positions
            V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X) - divergence_term
            X = X + V

            # Ensure particles stay within bounds
            X = jnp.clip(X, lb, ub)

            # Update individual best and global best
            current_val = jnp.array([self.penalized_obj(x) for x in X])
            improved = current_val < pbest_val

            # Update individual best values
            pbest = pbest.at[improved].set(X[improved])
            pbest_val = pbest_val.at[improved].set(current_val[improved])

            # Update global best values
            new_gbest_val = jnp.min(pbest_val)
            if new_gbest_val < gbest_val:
                gbest = pbest[jnp.argmin(pbest_val)].copy()
                gbest_val = new_gbest_val

            # Store the updated particles positions
            path[i].append(X.copy())
            path_gbest.append(gbest.copy())

            p_diversity = jnp.linalg.norm(jnp.std(X, axis=0))
            v_diversity = jnp.linalg.norm(jnp.std(V, axis=0))
            particle_diversity.append(p_diversity)
            velocity_diversity.append(v_diversity)

            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()  # Stop memory tracking

        return {
            "method_name": "Particle Swarm Optimization"
            if not self.f_cons
            else "Particle Swarm Optimization with Penalty",
            "x0": jnp.mean(X, axis=0),
            "xopt": gbest,
            "fmin": gbest_val,
            "num_iter": num_iter,
            "path_particles": path,
            "path": jnp.array(path_gbest),
            "time": elapsed_time,
            "memory_peak": peak / 1e6,  # Peak memory in MB
            "particle_diversity": particle_diversity,
            "velocity_diversity": velocity_diversity,
        }



def particle_swarm(
    divergence="baseline",
    bounds=[(-5, 5), (-5, 5)],  # noqa
    alpha=0.5,
    w=0.1,
    c1=0.25,
    c2=2,
    n_particles=30,
    **kwargs,
):
    """Particle Swarm Optimization algorithm."""
    optimizer = ParticleSwarmOptimization(**kwargs)
    result = optimizer.optimize(divergence, bounds, alpha, w, c1, c2, n_particles)

    return result
