import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

from optymus.methods.utils import BaseOptimizer


class StochasticGradientDescent(BaseOptimizer):
    def optimize(self, x0, data):
        start_time = time.time()
        x = x0.astype(float)

        path = [x]
        num_iter = 0

        progress_bar = tqdm(range(self.max_iter), desc=f"SGD {num_iter}") if self.verbose else range(self.max_iter)

        for _ in progress_bar:
            grad_sum = jnp.zeros_like(x)
            for i in range(0, len(data), self.batch_size):
                batch = data[i : i + self.batch_size]
                grad = jax.grad(self.penalized_obj)(x, batch)
                grad_sum += grad

            avg_grad = grad_sum / len(data)

            if jnp.linalg.norm(avg_grad) < self.tol:
                break

            x = x - self.learning_rate * avg_grad
            path.append(x)
            num_iter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        return {
            "method_name": "Stochastic Gradient Descent"
            if not self.f_cons
            else "Stochastic Gradient Descent with Penalty",
            "x0": x0,
            "xopt": x,
            "fmin": self.f_obj(x, data, *self.args),
            "num_iter": num_iter,
            "path": jnp.array(path),
            "time": elapsed_time,
        }


def sgd(**kwargs):
    optimizer = StochasticGradientDescent(**kwargs)
    return optimizer.optimize()
