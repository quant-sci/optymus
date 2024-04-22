
import jax
import jax.numpy as jnp
from optymus.utils import line_search

def newton_raphson(f_obj, x0, tol=1e-5, maxiter=100):
  """Newton-Raphson"""

  grad = jax.grad(f_obj)
  hess = jax.hessian(f_obj)

  x = x0.copy()
  path = [x]
  alphas = []
  num_iter = 0

  for _ in range(maxiter):
    g = grad(x)
    H = hess(x)

    if jnp.linalg.norm(g) < tol:
      break

    p = jax.scipy.linalg.solve(H, -g)

    ls_result = line_search(f_obj, x, p)
    alpha = ls_result['alpha']
    x = ls_result['xopt']

    alphas.append(alpha)
    path.append(x)
    num_iter += 1

  result = {
      'method_name': 'Newton-Raphson',
      'xopt': x,
      'fmin': f_obj(x),
      'num_iter': num_iter,
      'path': jnp.array(path),
      'alphas': jnp.array(alphas)
  }
  return result
