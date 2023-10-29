
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from qsopt.search import line_search
import numpy as np

def univariant(objective_function, initial_point, tol=1e-5, max_iter=100):
    x = initial_point.copy()
    n = len(x)                    
    df = np.zeros(n)                  
    u = np.identity(n)
    path = [x]
    alphas = []
    num_iter = 0       
    for _ in range(max_iter):            
        xOld = x.copy()           
        for i in range(n):
            v = u[i]
            r = line_search(objective_function, x, v)
            x = r['xopt']
            alphas.append(r['alpha'])
            path.append(x)
            if np.sqrt(np.dot(x-xOld, x-xOld)/n) < tol:
                break
            
            result = {
                'method_name': 'Univariant',
                'xopt': x, 
                'fmin': fLast, 
                'num_iter': num_iter, 
                'path': np.array(path),
                'alphas': np.array(alphas),
                }
            return result

def powell(objective_function, initial_point, tol=1e-5, max_iter=100):
    x = initial_point.copy()
    n = len(x)                    
    df = np.zeros(n)                  
    u = np.identity(n)
    path = [x]
    alphas = []
    num_iter = 0       
    for _ in range(max_iter):            
        xOld = x.copy()           
        fOld = objective_function(xOld)
        for i in range(n):
            v = u[i]
            r = line_search(objective_function, x, v)
            fMin = r['fmin']
            df[i] = fOld - fMin
            fOld = fMin
            alphas.append(r['alpha'])
            x = r['xopt']
        v = x - xOld
        r = line_search(objective_function, x, v)
        fLast = r['fmin']
        x = r['xopt']
        num_iter += 1
        if np.sqrt(np.dot(x-xOld, x-xOld)/n) < tol:
            result = {
                'method_name': 'Powell',
                'xopt': x, 
                'fmin': fLast, 
                'num_iter': num_iter, 
                'path': np.array(path),
                'alphas': np.array(alphas),
                }
            return result
        iMax = np.argmax(df)
        for i in range(iMax,n-1):
            u[i] = u[i+1]
        u[n-1] = v
        path.append(x)

def steepest_descent(objective_function, grad, initial_point, tol=1e-4, max_iter=100):
    x = initial_point.copy()
    g = grad(x)
    d = -g
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        r = line_search(objective_function, x, d)
        alphas.append(r['alpha'])
        x = r['xopt']
        g = grad(x)
        d = -g
        num_iter += 1
        path.append(x)
    result = {
                'method_name': 'Steepest Descent',
                'xopt': x, 
                'fmin': objective_function(x), 
                'num_iter': num_iter, 
                'path': np.array(path),
                'alphas': np.array(alphas),
                }        
    return result


def fletcher_reeves(objective_function, grad, initial_point, tol=1e-5, maxiter=100):
    x = initial_point.copy()
    gradient = grad(x)
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        if np.linalg.norm(gradient) <= tol:
            break
        direction = -gradient
        r = line_search(objective_function, x, direction)
        x = r['xopt']
        new_gradient = grad(x)
        if np.linalg.norm(new_gradient) <= tol:
            break
        beta = np.dot(new_gradient, new_gradient) / np.dot(gradient, gradient)
        direction = -new_gradient + beta * direction
        r = line_search(objective_function, x, direction)
        x = r['xopt']
        alphas.append(r['alpha'])
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'Fletcher-Reeves',
        'xopt': x,
        'fmin': objective_function(x),
        'num_iter': num_iter,
        'path': np.array(path),
        'alphas': np.array(alphas)
        }
    return result


def bfgs(objective_function, grad, initial_point, tol=1e-5, maxiter=100):
    x = initial_point.copy()
    H = np.eye(len(x))  # Initialize the Hessian approximation as the identity matrix
    path = [x]
    alphas = []
    num_iter = 0
    for i in range(maxiter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        p = -np.dot(H, g)
        ls = line_search(objective_function, x, p)
        x_new = ls['xopt']
        alphas.append(ls['alpha'])
        s = x_new - x
        y = grad(x_new) - g
        # BFGS update
        rho = 1.0 / np.dot(y, s)
        A = np.eye(len(x)) - rho * np.outer(s, y)
        B = np.eye(len(x)) - rho * np.outer(y, s)
        H = np.dot(np.dot(A, H), B) + rho * np.outer(s, s)
        x = x_new
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'BFGS',
        'xopt': x,
        'fmin': objective_function(x),
        'num_iter': num_iter,
        'initial_point': initial_point,
        'path': np.array(path)
        }
    return result

def newton_raphson(objective_function, gradient, hessian, initial_point, tol=1e-5, maxiter=100):
    x = initial_point.copy()
    path = [x]
    alphas = []
    num_iter = 0
    for _ in range(maxiter):
        g = gradient(x)
        H = hessian(x)  # Compute the Hessian at the current point x
        if np.linalg.norm(g) < tol:
            break
        p = np.linalg.solve(H, -g)
        ls = line_search(objective_function, x, p)
        alpha = ls['alpha']
        x = ls['xopt']
        alphas.append(alpha)
        path.append(x)
        num_iter += 1
    result = {
        'method_name': 'Newton-Raphson',
        'xopt': x,
        'fmin': objective_function(x),
        'num_iter': num_iter,
        'path': np.array(path),
        'alphas': np.array(alphas)
    }
    return result