import numpy as np

def bracket_minimum(f, x=0, s=0.01):
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    while True:
        c, yc = b + s, f(b + s)
        if yc > yb:
            return (a, c) if np.all(a < c) else (c, a)
        a, ya, b, yb = b, yb, c, yc

def golden_section(f, a, b, tol=1e-5):
    phi = (np.sqrt(5)-1)/2
    num_iter = 0
    beta = np.linalg.norm(b-a)
    alpha_e = a + (1 - phi)*beta
    alpha_d = a + (phi*beta)
    path = []
    
    while beta > tol:
        if f(alpha_e) < f(alpha_d):
            b = alpha_d
        else:
            a = alpha_e

        beta = np.linalg.norm(b-a)
        alpha_e = a + (1 - phi)*beta
        alpha_d = a + (phi*beta)

        num_iter += 1
        path.append((a+b)/2)
        
    alpha = (b + a) / 2
    fmin = f(alpha)

    result = {
        'method_name': 'Golden Search',
        'xopt': alpha,
        'fmin': fmin,
        'num_iter': num_iter,
        'path': np.array(path),
    }
    return result

import scipy.optimize as opt
def line_search(f, x, d, step_size=0.01):
    objective = lambda alpha: f(x + alpha * d)
    a, b = bracket_minimum(objective, s=step_size)
    alpha = opt.golden(objective, brack=(a, b), tol=1e-5)
    x_opt = x + alpha * d
    return {
        'method_name': 'Line Search',
        'alpha': alpha,
        'xopt': x_opt,
        'fmin': f(x_opt),
        'num_iter': 1,
        'function': f
    }