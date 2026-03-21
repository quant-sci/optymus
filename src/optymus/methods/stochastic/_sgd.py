import time

import jax
import jax.numpy as jnp
from rich.progress import track

from optymus.methods.utils import BaseOptimizer


class StochasticGradientDescent(BaseOptimizer):
    def optimize(self, x0, data):
        start_time = time.time()
        x = x0.astype(float)

        path = [x]
        f_history = [float(self.f_obj(x, data, *self.args))]
        grad_norms = []
        num_iter = 0
        termination_reason = "max_iter_reached"

        progress_bar = track(range(self.max_iter), description=f"SGD {num_iter}") if self.verbose else range(self.max_iter)

        for _ in progress_bar:
            grad_sum = jnp.zeros_like(x)
            for i in range(0, len(data), self.batch_size):
                batch = data[i : i + self.batch_size]
                grad = jax.grad(self.penalized_obj)(x, batch)
                grad_sum += grad

            avg_grad = grad_sum / len(data)
            grad_norms.append(float(jnp.linalg.norm(avg_grad)))

            if jnp.linalg.norm(avg_grad) < self.tol:
                termination_reason = "gradient_norm_below_tol"
                break

            x = x - self.learning_rate * avg_grad
            path.append(x)
            f_history.append(float(self.f_obj(x, data, *self.args)))
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
            "f_history": jnp.array(f_history),
            "grad_norms": jnp.array(grad_norms),
            "termination_reason": termination_reason,
            "time": elapsed_time,
        }


def sgd(**kwargs):
    optimizer = StochasticGradientDescent(**kwargs)
    return optimizer.optimize()
