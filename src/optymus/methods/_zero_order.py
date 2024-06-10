import jax
import jax.numpy as jnp

from optymus.search import line_search


def univariant(f_obj=None, f_constr=None, x0=None, tol=1e-5, max_iter=100):
    x = x0.copy()
    n = len(x)
    u = jnp.identity(n)
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(max_iter):
        for i in range(n):
            v = u[i]
            r = line_search(f_obj, x, v)
            x = r['xopt']
            alphas.append(r['alpha'])
            path.append(x)
            if jnp.linalg.norm(jax.grad(f_obj)(x)) < tol:
                return {
                    'method_name': 'Univariant',
                    'xopt': x,
                    'fmin': f_obj(x),
                    'num_iter': num_iter,
                    'path': jnp.array(path),
                    'alphas': jnp.array(alphas),
                    }
        num_iter += 1
    return None

def powell(f_obj=None, f_constr=None, x0=None, tol=1e-5, max_iter=100):
    """Powell's Method"""

    # Define gradient function using JAX's automatic differentiation
    grad = jax.grad(f_obj)

    # Function to create basis vectors
    def basis(i, n):
        return jnp.eye(n)[:, i-1]

    n = len(x0)
    u = [basis(i, n) for i in range(1, n+1)]  # Initial basis vectors
    x = x0.copy()
    path = [x]
    alphas = []
    num_iter = 0
    while jnp.linalg.norm(grad(x0)) > tol and num_iter < max_iter:
        x_prime = x.copy()
        for i in range(n):
            d = u[i]
            r = line_search(f_obj, x_prime, d)
            x_prime = r['xopt']
            alphas.append(r['alpha'])
            path.append(x_prime)

        # Update basis vectors
        for i in range(n-1):
            u[i] = u[i+1]
        u[n-1] = x_prime - x

        # Perform line search along the new direction
        d = u[n-1]
        r = line_search(f_obj, x, d)
        x_prime = r['xopt']

        x0 = x_prime
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1

        return {
            'method_name': 'Powell',
            'xopt': x,
            'fmin': f_obj(x),
            'num_iter': num_iter,
            'path': jnp.array(path),
            'alphas': jnp.array(alphas),
        }
    return None
