import numpy as np
from optymus.utils import line_search

def univariant(f_obj, x0, grad, tol=1e-5, max_iter=100):
    x = x0.copy()
    n = len(x)                        
    u = np.identity(n)
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
            if np.linalg.norm(grad(x)) < tol:
                result = {
                    'method_name': 'Univariant',
                    'xopt': x, 
                    'fmin': f_obj(x), 
                    'num_iter': num_iter, 
                    'path': np.array(path),
                    'alphas': np.array(alphas),
                    }
                return result
                
        num_iter += 1

def powell(f_obj, x0, grad, tol=1e-5, max_iter=100):
    x = x0.copy()
    def basis(i, n):
        return np.eye(n)[:, i-1]
    n = len(x)                   
    u = [basis(i,n) for i in range(1, n+1)]
    path = [x]
    alphas = []
    num_iter = 0       
    while np.linalg.norm(grad(x0)) > tol:            
        x_prime = x           
        for i in range(n):
            d = u[i]
            r = line_search(f_obj, x_prime, d)
            x_prime = r['xopt']
            alphas.append(r['alpha'])
            path.append(x_prime)
        for i in range(n-1):
            u[i] = u[i+1]
        u[n-1] = x_prime - x
        d = u[n-1]
        r = line_search(f_obj, x, d)
        x_prime = r['xopt']
        x0 = x_prime
        alphas.append(r['alpha'])
        path.append(x)
        fLast = f_obj(x)
        num_iter += 1
    result = {
            'method_name': 'Powell',
            'xopt': x, 
            'fmin': fLast, 
            'num_iter': num_iter, 
            'path': np.array(path),
            'alphas': np.array(alphas),
            }
    return result
