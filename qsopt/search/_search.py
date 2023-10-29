import numpy as np

def golden_search(f, lbound, ubound, max_iter=100, tol=1e-5):
    phi = (np.sqrt(5)-1)/2  # Golden ratio
    num_iter = 0
    a, b = lbound, ubound
    beta = np.linalg.norm(b-a)
    alpha_e = a + (1 - phi)*beta
    alpha_d = a + (phi*beta)
    a_list = [a]
    b_list = [b]
    
    while beta > tol:
        if num_iter > max_iter:
            break
        if f(alpha_e) < f(alpha_d):  # f(alpha) > f(alpha_d) to find the maximum
            b = alpha_d
        else:
            a = alpha_e

        beta = np.linalg.norm(b-a)
        alpha_e = a + (1 - phi)*beta
        alpha_d = a + (phi*beta)

        num_iter += 1
        a_list.append(a)
        b_list.append(b)
        
    alpha = (b + a) / 2
    fmin = f(alpha)

    result = {
        'alpha': alpha[np.argmin(alpha)],
        'fmin': fmin,
        'num_iter': num_iter,
        'path_a': np.array(a_list),
        'path_b': np.array(b_list),
    }
    return result

def line_search(objective_function, initial_point, d, search_method='golden_search'):
    def f(alpha): return objective_function(initial_point + alpha*d)
    if search_method == 'golden_search':
        a, b = np.array([-10, -10]), np.array([10, 10])
        r = golden_search(f, a, b)
        a = r['alpha']
        x_opt = initial_point + a*d
        result = {
            'alpha': a,
            'xopt': x_opt,
            'fmin': objective_function(x_opt),
            'initial_point': initial_point,
            'num_iter': r['num_iter'],
            'path_a': r['path_a'],
            'path_b': r['path_b'],
        }

    return result