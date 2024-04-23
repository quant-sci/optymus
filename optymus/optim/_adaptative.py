import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def adam(f_obj, x0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-5, max_iter=100):
    """Adam optimization algorithm"""

    grad = jax.grad(f_obj)

    x = x0.copy()
    m = jnp.zeros_like(x)  # First moment estimate
    v = jnp.zeros_like(x)  # Second moment estimate
    t = 0  # Time step
    path = [x]

    for _ in range(max_iter):
        t += 1
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g  # Update biased first moment estimate
        v = beta2 * v + (1 - beta2) * g**2  # Update biased second moment estimate
        m_hat = m / (1 - beta1**t)  # Compute bias-corrected first moment estimate
        v_hat = v / (1 - beta2**t)  # Compute bias-corrected second moment estimate
        x = x - alpha * m_hat / (jnp.sqrt(v_hat) + eps)  # Update parameters

        path.append(x)
        
        if jnp.linalg.norm(g) < tol:
            break

    result = {
        'method_name': 'Adam',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': t,
        'path': jnp.array(path)
    }
    return result

def adagrad(f_obj, x0, alpha=0.01, eps=1e-8, tol=1e-5, max_iter=100):
  """Adagrad optimizer"""

  grad = jax.grad(f_obj)
  x = x0.copy()
  g_sq_sum = jnp.zeros_like(x)
  path = [x]

  for _ in range(max_iter):
    g = grad(x)
    g_sq_sum += g**2
    x -= alpha * g / (jnp.sqrt(g_sq_sum) + eps)
    path.append(x)
    if jnp.linalg.norm(g) < tol:
      break

  return {'method_name': 'Adagrad', 
          'xopt': x, 
          'fmin': f_obj(x), 
          'num_iter': _, 
          'path': jnp.array(path)
          }

# --- RMSprop ---
def rmsprop(f_obj, x0, alpha=0.001, beta=0.9, eps=1e-8, tol=1e-5, max_iter=100):
  """RMSprop optimizer"""

  grad = jax.grad(f_obj)
  x = x0.copy()
  Eg2 = jnp.zeros_like(x)
  path = [x]

  for _ in range(max_iter):
    g = grad(x)
    Eg2 = beta * Eg2 + (1 - beta) * g**2
    x -= alpha * g / (jnp.sqrt(Eg2) + eps)
    path.append(x)
    if jnp.linalg.norm(g) < tol:
      break

  return {'method_name': 'RMSprop', 
          'xopt': x, 
          'fmin': f_obj(x), 
          'num_iter': _, 
          'path': jnp.array(path)
          }

def adamax(f_obj, x0, alpha=0.002, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-5, max_iter=100):
  """Adamax optimizer"""

  grad = jax.grad(f_obj)
  x = x0.copy()
  m = jnp.zeros_like(x)
  u = jnp.zeros_like(x)
  path = [x]

  for t in range(1, max_iter + 1):
    g = grad(x)
    m = beta1 * m + (1 - beta1) * g
    u = jnp.maximum(beta2 * u, jnp.abs(g))
    x -= alpha * m / (u + eps)
    path.append(x)
    if jnp.linalg.norm(g) < tol:
      break

  return {'method_name': 'Adamax', 
          'xopt': x, 
          'fmin': f_obj(x), 
          'num_iter': t, 
          'path': jnp.array(path)
          }

def yogi(f_obj, x0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-3, tol=1e-5, max_iter=100):
  """Yogi optimizer"""

  grad = jax.grad(f_obj)
  x = x0.copy()
  m = jnp.zeros_like(x)
  v = jnp.zeros_like(x)
  sign_g = jnp.zeros_like(x)
  path = [x]

  for t in range(1, max_iter + 1):
    g = grad(x)
    sign_g = jnp.sign(g)
    m = beta1 * m + (1 - beta1) * sign_g * g
    v = beta2 * v + (1 - beta2) * g**2
    x -= alpha * m / (jnp.sqrt(v) + eps)
    path.append(x)
    if jnp.linalg.norm(g) < tol:
      break

  return {'method_name': 'Yogi', 
          'xopt': x, 
          'fmin': f_obj(x), 
          'num_iter': t, 
          'path': jnp.array(path)
          }
