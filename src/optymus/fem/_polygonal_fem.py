"""
Polygonal Finite Element Method for 2D Plane Stress.

This module implements a minimal FEM solver for polygonal meshes,
designed to work with the polymesher output for topology optimization.

Based on the PolyTop approach (Talischi et al., 2012).
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def plane_stress_D(E, nu):
    """
    Compute plane stress constitutive matrix.

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    D : ndarray (3, 3)
        Constitutive matrix for plane stress
    """
    factor = E / (1 - nu**2)
    D = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1 - nu) / 2.0]
    ])
    return D


def polygon_area(vertices):
    """
    Compute polygon area using the shoelace formula.

    Parameters
    ----------
    vertices : ndarray (n, 2)
        Polygon vertex coordinates

    Returns
    -------
    area : float
        Polygon area (always positive)
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def polygon_centroid(vertices):
    """
    Compute polygon centroid.

    Parameters
    ----------
    vertices : ndarray (n, 2)
        Polygon vertex coordinates

    Returns
    -------
    centroid : ndarray (2,)
        Centroid coordinates
    """
    n = len(vertices)
    cx, cy = 0.0, 0.0
    signed_area = 0.0

    for i in range(n):
        j = (i + 1) % n
        cross = vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        signed_area += cross
        cx += (vertices[i, 0] + vertices[j, 0]) * cross
        cy += (vertices[i, 1] + vertices[j, 1]) * cross

    signed_area /= 2.0
    if abs(signed_area) < 1e-12:
        return np.mean(vertices, axis=0)

    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    return np.array([cx, cy])


def triangle_B_matrix(tri_coords):
    """
    Compute B matrix (strain-displacement) for a linear triangle.

    Parameters
    ----------
    tri_coords : ndarray (3, 2)
        Triangle vertex coordinates

    Returns
    -------
    B : ndarray (3, 6)
        Strain-displacement matrix
    area : float
        Triangle area
    """
    x1, y1 = tri_coords[0]
    x2, y2 = tri_coords[1]
    x3, y3 = tri_coords[2]

    # Area using cross product
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    if area < 1e-12:
        return np.zeros((3, 6)), 1e-12

    # Shape function derivatives
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    B = (1.0 / (2.0 * area)) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3]
    ])

    return B, area


def polygon_stiffness_matrix(node_coords, D, thickness=1.0):
    """
    Compute element stiffness matrix for a polygonal element.

    Uses fan triangulation with centroid as internal node, then
    static condensation to remove the centroid DOFs.

    Parameters
    ----------
    node_coords : ndarray (n_nodes, 2)
        Coordinates of element vertices (in order)
    D : ndarray (3, 3)
        Constitutive matrix
    thickness : float
        Element thickness

    Returns
    -------
    Ke : ndarray (2*n_nodes, 2*n_nodes)
        Element stiffness matrix
    """
    n_nodes = len(node_coords)

    # For triangles, use direct computation
    if n_nodes == 3:
        B, area = triangle_B_matrix(node_coords)
        if area < 1e-12:
            return np.zeros((6, 6))
        return thickness * area * (B.T @ D @ B)

    # For polygons: use centroid + static condensation
    centroid = polygon_centroid(node_coords)

    # Build expanded stiffness matrix including centroid
    # DOF ordering: [centroid_x, centroid_y, node0_x, node0_y, node1_x, ...]
    n_total_dof = 2 * (n_nodes + 1)  # Including centroid
    K_expanded = np.zeros((n_total_dof, n_total_dof))

    for i in range(n_nodes):
        j = (i + 1) % n_nodes

        # Triangle: centroid (idx 0), node_i (idx i+1), node_j (idx j+1)
        tri_coords = np.array([centroid, node_coords[i], node_coords[j]])
        B_tri, area_tri = triangle_B_matrix(tri_coords)

        if area_tri < 1e-12:
            continue

        # Triangle stiffness
        Ke_tri = thickness * area_tri * (B_tri.T @ D @ B_tri)

        # Map to expanded system
        # Triangle local DOFs: 0,1 (centroid), 2,3 (node i), 4,5 (node j)
        # Expanded global DOFs: 0,1 (centroid), 2+2*i, 3+2*i (node i), 2+2*j, 3+2*j (node j)
        local_to_global = [0, 1, 2 + 2*i, 3 + 2*i, 2 + 2*j, 3 + 2*j]

        for li in range(6):
            for lj in range(6):
                gi = local_to_global[li]
                gj = local_to_global[lj]
                K_expanded[gi, gj] += Ke_tri[li, lj]

    # Static condensation: remove centroid DOFs (first 2)
    # K = [Kcc  Kcb]    where c = centroid, b = boundary (polygon vertices)
    #     [Kbc  Kbb]
    # Condensed: K_condensed = Kbb - Kbc @ inv(Kcc) @ Kcb

    Kcc = K_expanded[0:2, 0:2]
    Kcb = K_expanded[0:2, 2:]
    Kbc = K_expanded[2:, 0:2]
    Kbb = K_expanded[2:, 2:]

    # Regularize Kcc if needed
    diag_kcc = np.diag(Kcc)
    if np.any(diag_kcc < 1e-12):
        Kcc = Kcc + 1e-10 * np.eye(2)

    try:
        Kcc_inv = np.linalg.inv(Kcc)
        Ke = Kbb - Kbc @ Kcc_inv @ Kcb
    except np.linalg.LinAlgError:
        # Fallback: just use Kbb
        Ke = Kbb

    # Ensure symmetry
    Ke = 0.5 * (Ke + Ke.T)

    return Ke


def assemble_global_stiffness(nodes, elements, D, thickness=1.0):
    """
    Assemble global stiffness matrix from element contributions.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 2)
        Global node coordinates
    elements : list of ndarray
        Element connectivity (list of node index arrays)
    D : ndarray (3, 3)
        Constitutive matrix
    thickness : float
        Element thickness

    Returns
    -------
    K : sparse matrix (2*n_nodes, 2*n_nodes)
        Global stiffness matrix in CSR format
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K = lil_matrix((n_dof, n_dof))

    for elem_nodes in elements:
        elem_coords = nodes[elem_nodes]
        Ke = polygon_stiffness_matrix(elem_coords, D, thickness)
        n_elem_nodes = len(elem_nodes)

        # Assembly
        for i_local in range(n_elem_nodes):
            i_global = elem_nodes[i_local]
            for j_local in range(n_elem_nodes):
                j_global = elem_nodes[j_local]
                for di in range(2):
                    for dj in range(2):
                        i_dof = 2 * i_global + di
                        j_dof = 2 * j_global + dj
                        K[i_dof, j_dof] += Ke[2*i_local + di, 2*j_local + dj]

    return K.tocsr()


def apply_loads(n_nodes, boundary_load):
    """
    Create global force vector from boundary loads.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    boundary_load : ndarray (n_load, 3)
        Load conditions [node_idx, load_x, load_y]

    Returns
    -------
    F : ndarray (2*n_nodes,)
        Global force vector
    """
    F = np.zeros(2 * n_nodes)

    for load in boundary_load:
        node_idx = int(load[0])
        F[2 * node_idx] = load[1]       # x-component
        F[2 * node_idx + 1] = load[2]   # y-component

    return F


def apply_boundary_conditions(K, F, boundary_supp):
    """
    Apply Dirichlet boundary conditions using the penalty method.

    Parameters
    ----------
    K : sparse matrix
        Global stiffness matrix
    F : ndarray
        Global force vector
    boundary_supp : ndarray (n_supp, 3)
        Support conditions [node_idx, fix_x, fix_y]

    Returns
    -------
    K_mod : sparse matrix
        Modified stiffness matrix
    F_mod : ndarray
        Modified force vector
    """
    K_mod = K.tolil()
    F_mod = F.copy()

    # Compute penalty factor
    diag = K.diagonal()
    max_diag = np.max(np.abs(diag[diag != 0])) if np.any(diag != 0) else 1.0
    penalty = 1e10 * max_diag

    for supp in boundary_supp:
        node_idx = int(supp[0])
        fix_x, fix_y = int(supp[1]), int(supp[2])

        if fix_x:
            dof = 2 * node_idx
            K_mod[dof, dof] += penalty
            F_mod[dof] = 0.0

        if fix_y:
            dof = 2 * node_idx + 1
            K_mod[dof, dof] += penalty
            F_mod[dof] = 0.0

    return K_mod.tocsr(), F_mod


def solve_fem(nodes, elements, boundary_supp, boundary_load,
              E=1.0, nu=0.3, thickness=1.0):
    """
    Solve the FEM problem for a polygonal mesh.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 2)
        Node coordinates
    elements : list of ndarray
        Element connectivity (list of node index arrays)
    boundary_supp : ndarray
        Support boundary conditions [node_idx, fix_x, fix_y]
    boundary_load : ndarray
        Load boundary conditions [node_idx, load_x, load_y]
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    thickness : float
        Element thickness

    Returns
    -------
    dict
        Solution dictionary containing:
        - displacements: nodal displacement vector
        - compliance: structural compliance (f^T * u)
        - stiffness_matrix: global stiffness matrix (before BCs)
        - force_vector: global force vector
    """
    n_nodes = len(nodes)
    D = plane_stress_D(E, nu)

    # Assemble global stiffness
    K = assemble_global_stiffness(nodes, elements, D, thickness)

    # Create force vector
    F = apply_loads(n_nodes, boundary_load)

    # Apply boundary conditions
    K_bc, F_bc = apply_boundary_conditions(K, F, boundary_supp)

    # Solve the system
    U = spsolve(K_bc, F_bc)

    # Compute compliance
    compliance = float(F @ U)

    return {
        "displacements": U,
        "compliance": compliance,
        "stiffness_matrix": K,
        "force_vector": F
    }


def compute_element_compliance(nodes, elements, U, D, thickness=1.0):
    """
    Compute compliance contribution of each element.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 2)
        Node coordinates
    elements : list of ndarray
        Element connectivity
    U : ndarray
        Global displacement vector
    D : ndarray (3, 3)
        Constitutive matrix
    thickness : float
        Element thickness

    Returns
    -------
    ce : ndarray (n_elements,)
        Element-wise compliance values
    """
    n_elements = len(elements)
    ce = np.zeros(n_elements)

    for e, elem_nodes in enumerate(elements):
        elem_coords = nodes[elem_nodes]
        Ke = polygon_stiffness_matrix(elem_coords, D, thickness)

        # Extract element displacements
        elem_dofs = []
        for node in elem_nodes:
            elem_dofs.extend([2*node, 2*node + 1])
        u_e = U[elem_dofs]

        # Element compliance: u_e^T * Ke * u_e
        ce[e] = float(u_e @ Ke @ u_e)

    return ce
