import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def polytop(fem, opt):
    iter = 0
    tol = opt['tol'] * (opt['z_max'] - opt['z_min'])
    change = 2 * tol
    z = opt['z_ini']
    p = opt['p']
    
    e, de_dy, v, dv_dy = opt['mat_int_fnc'](np.dot(p, z))
    fig_handle, fig_data = initial_plot(fem, v)
    
    while (iter < opt['max_iter']) and (change > tol):
        iter += 1
        # Compute cost functionals and analysis sensitivities
        f, df_de, df_dv, fem = objective_fnc(fem, e, v)
        g, dg_de, dg_dv, fem = constraint_fnc(fem, e, v, opt['vol_frac'])
        
        # Compute design sensitivities
        df_dz = np.dot(p.T, de_dy * df_de + dv_dy * df_dv)
        dg_dz = np.dot(p.T, de_dy * dg_de + dv_dy * dg_dv)
        
        # Update design variable and analysis parameters
        z, change = update_scheme(df_dz, g, dg_dz, z, opt)
        e, de_dy, v, dv_dy = opt['mat_int_fnc'](np.dot(p, z))
        
        # Output results
        print(f'It: {iter} \t Objective: {f:.3f}\tChange: {change:.3f}')
        fig_handle.set_array(1 - v[fig_data])
        plt.draw()
    
    return z, v, fem

def objective_fnc(fem, e, v):
    u, fem = fe_analysis(fem, e)
    f = np.dot(fem['f'], u)
    temp = np.cumsum(-u[fem['i']] * fem['k'] * u[fem['j']])
    temp = temp[np.cumsum(fem['elem_n_dof']**2) - 1]
    df_de = np.diff(np.concatenate(([0], temp)))
    df_dv = np.zeros_like(v)
    return f, df_de, df_dv, fem

def constraint_fnc(fem, e, v, vol_frac):
    if 'elem_area' not in fem:
        fem['elem_area'] = np.zeros(fem['n_elem'])
        for el in range(fem['n_elem']):
            vx = fem['node'][fem['element'][el], 0]
            vy = fem['node'][fem['element'][el], 1]
            fem['elem_area'][el] = 0.5 * np.sum(vx * np.roll(vy, -1) - vy * np.roll(vx, -1))
    
    g = np.sum(fem['elem_area'] * v) / np.sum(fem['elem_area']) - vol_frac
    dg_de = np.zeros_like(e)
    dg_dv = fem['elem_area'] / np.sum(fem['elem_area'])
    return g, dg_de, dg_dv, fem

def update_scheme(df_dz, g, dg_dz, z0, opt):
    z_min, z_max = opt['z_min'], opt['z_max']
    move = opt['oc_move'] * (z_max - z_min)
    eta = opt['oc_eta']
    l1, l2 = 0, 1e6
    
    while l2 - l1 > 1e-4:
        l_mid = 0.5 * (l1 + l2)
        b = -(df_dz / dg_dz) / l_mid
        z_cnd = z_min + (z0 - z_min) * b**eta
        z_new = np.clip(z_cnd, z0 - move, z0 + move)
        z_new = np.clip(z_new, z_min, z_max)
        
        if g + np.dot(dg_dz, z_new - z0) > 0:
            l1 = l_mid
        else:
            l2 = l_mid
    
    change = np.max(np.abs(z_new - z0)) / (z_max - z_min)
    return z_new, change

def fe_analysis(fem, e):
    if 'k' not in fem:
        fem['elem_n_dof'] = np.array([2 * len(elem) for elem in fem['element']])
        n_dof_total = np.sum(fem['elem_n_dof']**2)
        fem['i'] = np.zeros(n_dof_total, dtype=int)
        fem['j'] = np.zeros(n_dof_total, dtype=int)
        fem['k'] = np.zeros(n_dof_total)
        fem['e'] = np.zeros(n_dof_total, dtype=int)
        
        index = 0
        if 'shape_fnc' not in fem:
            fem = tab_shape_fnc(fem)
        
        for el in range(fem['n_elem']):
            if not fem['reg'] or el == 0:
                ke = local_k(fem, fem['element'][el])
            n_dof = fem['elem_n_dof'][el]
            e_dof = np.array([[2*node-1, 2*node] for node in fem['element'][el]]).flatten()
            i, j = np.meshgrid(e_dof, e_dof)
            fem['i'][index:index+n_dof**2] = i.flatten()
            fem['j'][index:index+n_dof**2] = j.flatten()
            fem['k'][index:index+n_dof**2] = ke.flatten()
            fem['e'][index:index+n_dof**2] = el
            index += n_dof**2
        
        fem['f'] = np.zeros(2 * fem['n_node'])
        fem['f'][2*fem['load'][:, 0]-2] = fem['load'][:, 1]  # x-coordinate
        fem['f'][2*fem['load'][:, 0]-1] = fem['load'][:, 2]  # y-coordinate
        
        fixed_dofs = np.concatenate([
            fem['supp'][:, 1] * (2 * fem['supp'][:, 0] - 1),
            fem['supp'][:, 2] * (2 * fem['supp'][:, 0])
        ])
        fixed_dofs = fixed_dofs[fixed_dofs > 0]
        all_dofs = np.arange(1, 2*fem['n_node']+1)
        fem['free_dofs'] = np.setdiff1d(all_dofs, fixed_dofs)
    
    k = csr_matrix((e[fem['e']] * fem['k'], (fem['i']-1, fem['j']-1)), shape=(2*fem['n_node'], 2*fem['n_node']))
    k = (k + k.T) / 2
    u = np.zeros(2 * fem['n_node'])
    u[fem['free_dofs']-1] = spsolve(k[fem['free_dofs']-1][:, fem['free_dofs']-1], fem['f'][fem['free_dofs']-1])
    return u, fem

def local_k(fem, e_node):
    d = fem['e0'] / (1 - fem['nu0']**2) * np.array([
        [1, fem['nu0'], 0],
        [fem['nu0'], 1, 0],
        [0, 0, (1 - fem['nu0'])/2]
    ])
    nn = len(e_node)
    ke = np.zeros((2*nn, 2*nn))
    w = fem['shape_fnc'][nn]['w']
    
    for q in range(len(w)):
        dn_dxi = fem['shape_fnc'][nn]['dn_dxi'][:, :, q]
        j0 = np.dot(fem['node'][e_node].T, dn_dxi)
        dn_dx = np.linalg.solve(j0.T, dn_dxi.T).T
        b = np.zeros((3, 2*nn))
        b[0, 0::2] = dn_dx[:, 0]
        b[1, 1::2] = dn_dx[:, 1]
        b[2, 0::2] = dn_dx[:, 1]
        b[2, 1::2] = dn_dx[:, 0]
        ke += np.dot(np.dot(b.T, d), b) * w[q] * np.linalg.det(j0)
    
    return ke

def tab_shape_fnc(fem):
    elem_n_node = np.array([len(elem) for elem in fem['element']])
    fem['shape_fnc'] = [None] * (np.max(elem_n_node) + 1)
    
    for nn in range(np.min(elem_n_node), np.max(elem_n_node) + 1):
        w, q = poly_quad(nn)
        n = np.zeros((nn, 1, len(w)))
        dn_dxi = np.zeros((nn, 2, len(w)))
        
        for i in range(len(w)):
            n[:, :, i], dn_dxi[:, :, i] = poly_shape_fnc(nn, q[i])
        
        fem['shape_fnc'][nn] = {'w': w, 'n': n, 'dn_dxi': dn_dxi}
    
    return fem

def poly_shape_fnc(nn, xi):
    n = np.zeros(nn)
    alpha = np.zeros(nn)
    dn_dxi = np.zeros((nn, 2))
    dalpha = np.zeros((nn, 2))
    sum_alpha = 0.0
    sum_dalpha = np.zeros(2)
    a = np.zeros(nn)
    da = np.zeros((nn, 2))
    
    p, tri = poly_trnglt(nn, xi)
    
    for i in range(nn):
        sctr = tri[i]
        p_t = p[sctr]
        a[i] = 0.5 * np.linalg.det(np.column_stack((p_t, np.ones(3))))
        da[i, 0] = 0.5 * (p_t[2, 1] - p_t[1, 1])
        da[i, 1] = 0.5 * (p_t[1, 0] - p_t[2, 0])
    
    a = np.roll(a, 1)
    da = np.roll(da, 1, axis=0)
    
    for i in range(nn):
        alpha[i] = 1 / (a[i] * a[(i+1) % nn])
        dalpha[i] = -alpha[i] * (da[i] / a[i] + da[(i+1) % nn] / a[(i+1) % nn])
        sum_alpha += alpha[i]
        sum_dalpha += dalpha[i]
    
    for i in range(nn):
        n[i] = alpha[i] / sum_alpha
        dn_dxi[i] = (dalpha[i] - n[i] * sum_dalpha) / sum_alpha
    
    return n, dn_dxi

def poly_trnglt(nn, xi):
    p = np.column_stack((
        np.cos(2 * np.pi * np.arange(nn) / nn),
        np.sin(2 * np.pi * np.arange(nn) / nn)
    ))
    p = np.vstack((p, xi))
    tri = np.column_stack((
        np.full(nn, nn),
        np.arange(nn),
        np.roll(np.arange(nn), -1)
    ))
    return p, tri

def poly_quad(nn):
    w, q = tri_quad()
    p, tri = poly_trnglt(nn, np.array([0, 0]))
    point = np.zeros((nn * len(w), 2))
    weight = np.zeros(nn * len(w))
    
    for k in range(nn):
        sctr = tri[k]
        for i, wi in enumerate(w):
            n, dn_ds = tri_shape_fnc(q[i])
            j0 = np.dot(p[sctr].T, dn_ds)
            l = k * len(w) + i
            point[l] = np.dot(n, p[sctr])
            weight[l] = np.linalg.det(j0) * wi
    
    return weight, point

def tri_quad():
    point = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    weight = np.array([1/6, 1/6, 1/6])
    return weight, point

def tri_shape_fnc(s):
    n = np.array([1 - s[0] - s[1], s[0], s[1]])
    dn_ds = np.array([[-1, -1], [1, 0], [0, 1]])
    return n, dn_ds

def initial_plot(fem, z0):
    tri = []
    map_data = []
    for el, element in enumerate(fem['element']):
        for enode in range(len(element) - 2):
            tri.append([element[0], element[enode+1], element[enode+2]])
            map_data.append(el)
    
    tri = np.array(tri) - 1  # Adjust for 0-based indexing
    map_data = np.array(map_data)
    
    fig, ax = plt.subplots()
    handle = ax.tripcolor(fem['node'][:, 0], fem['node'][:, 1], tri, 1 - z0[map_data], 
                          shading='flat', cmap='gray')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.colorbar(handle)
    plt.draw()
    
    return handle, map_data