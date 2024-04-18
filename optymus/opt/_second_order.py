
import numpy as np
from optymus.utils import line_search

def newton_raphson(f_obj, x0, grad, hess, tol=1e-5, maxiter=100):
    x = x0.copy()
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        g = grad(x)
        H = hess(x)  # Compute the hess at the current point x
        if np.linalg.norm(g) < tol:
            break
        p = np.linalg.solve(H, -g)
        ls = line_search(f_obj, x, p)
        alpha = ls['alpha']
        x = ls['xopt']
        alphas.append(alpha)
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'Newton-Raphson',
        'xopt': x,
        'fmin': f_obj(x),
        'num_iter': num_iter,
        'path': np.array(path),
        'alphas': np.array(alphas)
    }
    return result