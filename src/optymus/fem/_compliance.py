"""
Compliance Objective Function for Topology Optimization.

This module provides the SIMP (Solid Isotropic Material with Penalization)
compliance objective function that is compatible with the optymus Optimizer.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from ._polygonal_fem import (
    plane_stress_D,
    polygon_area,
    polygon_stiffness_matrix,
    apply_loads,
    apply_boundary_conditions,
)


class ComplianceObjective:
    """
    Structural compliance objective function for topology optimization.

    Implements SIMP (Solid Isotropic Material with Penalization) method
    for density-based topology optimization.

    Parameters
    ----------
    mesh : dict
        Output from polymesher containing:
        - node: ndarray (n_nodes, 2)
        - element: list of ndarray
        - boundary_supp: ndarray (n_supp, 3)
        - boundary_load: ndarray (n_load, 3)
    E0 : float
        Young's modulus of solid material
    Emin : float
        Minimum Young's modulus (for numerical stability)
    nu : float
        Poisson's ratio
    penal : float
        SIMP penalization power (typically 3)
    volume_fraction : float
        Target volume fraction constraint
    thickness : float
        Element thickness
    filter_radius : float, optional
        Density filter radius. If None, no filtering is applied.
        Recommended: 1.5 * sqrt(total_area / n_elements)
    """

    def __init__(self, mesh, E0=1.0, Emin=1e-9, nu=0.3, penal=3.0,
                 volume_fraction=0.5, thickness=1.0, filter_radius=None):
        self.nodes = np.asarray(mesh["node"])
        self.elements = mesh["element"]
        self.boundary_supp = np.asarray(mesh["boundary_supp"])
        self.boundary_load = np.asarray(mesh["boundary_load"])

        self.E0 = E0
        self.Emin = Emin
        self.nu = nu
        self.penal = penal
        self.volume_fraction = volume_fraction
        self.thickness = thickness

        self.n_elements = len(self.elements)
        self.n_nodes = len(self.nodes)
        self.filter_radius = filter_radius

        # Precompute element stiffness matrices (for unit Young's modulus)
        self._precompute_element_stiffness()

        # Compute element volumes/areas
        self._compute_element_volumes()

        # Setup DOF mapping for assembly
        self._setup_dof_mapping()

        # Build density filter matrix if filter_radius is specified
        self._build_filter_matrix()

        # Cache for last solution (for gradient computation)
        self._last_U = None
        self._last_densities = None

    def _precompute_element_stiffness(self):
        """Precompute element stiffness matrices with E=1."""
        D_unit = plane_stress_D(1.0, self.nu)
        self.Ke_list = []

        for elem_nodes in self.elements:
            elem_coords = self.nodes[elem_nodes]
            Ke = polygon_stiffness_matrix(elem_coords, D_unit, self.thickness)
            self.Ke_list.append(Ke)

    def _compute_element_volumes(self):
        """Compute element areas/volumes."""
        self.element_volumes = np.zeros(self.n_elements)
        for i, elem_nodes in enumerate(self.elements):
            elem_coords = self.nodes[elem_nodes]
            self.element_volumes[i] = polygon_area(elem_coords)

        self.total_volume = np.sum(self.element_volumes)

    def _setup_dof_mapping(self):
        """Setup DOF mapping for each element."""
        self.element_dofs = []
        for elem_nodes in self.elements:
            dofs = []
            for node in elem_nodes:
                dofs.extend([2*node, 2*node + 1])
            self.element_dofs.append(np.array(dofs))

    def _compute_element_centroids(self):
        """Compute centroid of each element."""
        centroids = np.zeros((self.n_elements, 2))
        for i, elem_nodes in enumerate(self.elements):
            elem_coords = self.nodes[elem_nodes]
            centroids[i] = np.mean(elem_coords, axis=0)
        return centroids

    def _build_filter_matrix(self):
        """
        Build the density filter matrix based on element centroid distances.

        Filter weight: H_ij = max(0, rmin - dist(i,j))
        """
        if self.filter_radius is None or self.filter_radius <= 0:
            self.H = None
            self.Hs = None
            return

        rmin = self.filter_radius
        centroids = self._compute_element_centroids()

        # Build sparse filter matrix
        H = lil_matrix((self.n_elements, self.n_elements))

        for i in range(self.n_elements):
            for j in range(self.n_elements):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < rmin:
                    H[i, j] = rmin - dist

        self.H = H.tocsr()
        # Precompute sum for normalization: Hs[i] = sum_j(H_ij * v_j)
        self.Hs = np.array(self.H @ self.element_volumes).flatten()

    def filter_densities(self, x):
        """
        Apply density filter to element densities.

        x_filtered[i] = sum_j(H_ij * v_j * x_j) / sum_j(H_ij * v_j)

        Parameters
        ----------
        x : ndarray
            Element densities

        Returns
        -------
        x_filtered : ndarray
            Filtered densities
        """
        if self.H is None:
            return x
        return np.array(self.H @ (x * self.element_volumes)).flatten() / self.Hs

    def filter_sensitivities(self, x, dc):
        """
        Apply sensitivity filter.

        dc_filtered[i] = sum_j(H_ij * x_j * dc_j) / (x_i * sum_j(H_ij))

        Parameters
        ----------
        x : ndarray
            Element densities
        dc : ndarray
            Raw sensitivities

        Returns
        -------
        dc_filtered : ndarray
            Filtered sensitivities
        """
        if self.H is None:
            return dc
        H_sum = np.array(self.H.sum(axis=1)).flatten()
        return np.array(self.H @ (x * dc)).flatten() / (np.maximum(x, 1e-3) * H_sum)

    def _simp_stiffness(self, rho):
        """
        SIMP interpolation for element stiffness.

        E(rho) = Emin + rho^p * (E0 - Emin)

        Parameters
        ----------
        rho : float
            Element density

        Returns
        -------
        E : float
            Interpolated Young's modulus
        """
        return self.Emin + (rho ** self.penal) * (self.E0 - self.Emin)

    def _assemble_stiffness(self, densities):
        """
        Assemble global stiffness matrix with SIMP interpolation.

        Parameters
        ----------
        densities : ndarray (n_elements,)
            Element densities in [0, 1]

        Returns
        -------
        K : sparse matrix
            Global stiffness matrix
        """
        n_dof = 2 * self.n_nodes
        K = lil_matrix((n_dof, n_dof))

        for e, (elem_nodes, Ke, elem_dofs) in enumerate(zip(
                self.elements, self.Ke_list, self.element_dofs)):

            # SIMP interpolation
            E_e = self._simp_stiffness(densities[e])

            # Scale element stiffness
            Ke_scaled = E_e * Ke

            # Assemble
            n_elem_dof = len(elem_dofs)
            for i in range(n_elem_dof):
                for j in range(n_elem_dof):
                    K[elem_dofs[i], elem_dofs[j]] += Ke_scaled[i, j]

        return K.tocsr()

    def _solve(self, densities):
        """
        Solve the FEM problem for given densities.

        Parameters
        ----------
        densities : ndarray (n_elements,)
            Element densities

        Returns
        -------
        U : ndarray
            Displacement vector
        """
        # Assemble stiffness
        K = self._assemble_stiffness(densities)

        # Apply loads and BCs
        F = apply_loads(self.n_nodes, self.boundary_load)
        K_bc, F_bc = apply_boundary_conditions(K, F, self.boundary_supp)

        # Solve
        U = spsolve(K_bc, F_bc)

        return U

    def __call__(self, densities):
        """
        Compute compliance objective value.

        Parameters
        ----------
        densities : ndarray (n_elements,)
            Element densities in [0, 1]

        Returns
        -------
        compliance : float
            Structural compliance (to be minimized)
        """
        densities = np.asarray(densities).flatten()
        densities = np.clip(densities, 1e-6, 1.0)

        # Solve for displacements
        U = self._solve(densities)

        # Cache for gradient computation
        self._last_U = U
        self._last_densities = densities.copy()

        # Compute compliance: C = F^T * U
        F = apply_loads(self.n_nodes, self.boundary_load)
        compliance = float(F @ U)

        return compliance

    def gradient(self, densities):
        """
        Compute compliance gradient (sensitivity analysis).

        dC/drho_e = -p * rho_e^(p-1) * (E0 - Emin) * u_e^T * Ke * u_e

        Parameters
        ----------
        densities : ndarray (n_elements,)
            Element densities

        Returns
        -------
        dc : ndarray (n_elements,)
            Compliance gradient with respect to densities
        """
        densities = np.asarray(densities).flatten()
        densities = np.clip(densities, 1e-6, 1.0)

        # Use cached solution if available and densities match
        if (self._last_U is not None and
            self._last_densities is not None and
            np.allclose(densities, self._last_densities)):
            U = self._last_U
        else:
            U = self._solve(densities)
            self._last_U = U
            self._last_densities = densities.copy()

        # Compute element-wise sensitivities
        dc = np.zeros(self.n_elements)

        for e, (Ke, elem_dofs) in enumerate(zip(self.Ke_list, self.element_dofs)):
            u_e = U[elem_dofs]

            # Sensitivity: dC/drho = -p * rho^(p-1) * (E0 - Emin) * u^T * Ke * u
            dc[e] = -self.penal * (densities[e] ** (self.penal - 1)) * \
                    (self.E0 - self.Emin) * float(u_e @ Ke @ u_e)

        return dc

    def volume_constraint(self, densities):
        """
        Volume constraint function.

        g(x) = sum(rho * v) / V_total - vf

        Returns g <= 0 when feasible.

        Parameters
        ----------
        densities : ndarray (n_elements,)
            Element densities

        Returns
        -------
        g : float
            Constraint value (negative = feasible)
        """
        densities = np.asarray(densities).flatten()
        current_volume = np.sum(densities * self.element_volumes)
        target_volume = self.volume_fraction * self.total_volume
        return (current_volume - target_volume) / self.total_volume

    def volume_constraint_gradient(self, densities):
        """
        Gradient of volume constraint.

        dg/drho_e = v_e / V_total

        Parameters
        ----------
        densities : ndarray (n_elements,)
            Element densities

        Returns
        -------
        dg : ndarray (n_elements,)
            Volume constraint gradient
        """
        return self.element_volumes / self.total_volume

    def get_element_compliance(self, densities):
        """
        Compute element-wise compliance values.

        Parameters
        ----------
        densities : ndarray (n_elements,)
            Element densities

        Returns
        -------
        ce : ndarray (n_elements,)
            Element compliance values
        """
        densities = np.asarray(densities).flatten()
        densities = np.clip(densities, 1e-6, 1.0)

        if (self._last_U is not None and
            self._last_densities is not None and
            np.allclose(densities, self._last_densities)):
            U = self._last_U
        else:
            U = self._solve(densities)

        ce = np.zeros(self.n_elements)
        for e, (Ke, elem_dofs) in enumerate(zip(self.Ke_list, self.element_dofs)):
            E_e = self._simp_stiffness(densities[e])
            u_e = U[elem_dofs]
            ce[e] = E_e * float(u_e @ Ke @ u_e)

        return ce


def create_compliance_objective(mesh, **kwargs):
    """
    Factory function to create compliance objective from polymesher output.

    Parameters
    ----------
    mesh : dict
        Output from polymesher
    **kwargs
        Additional arguments for ComplianceObjective

    Returns
    -------
    ComplianceObjective
        Objective function compatible with optymus Optimizer
    """
    return ComplianceObjective(mesh, **kwargs)
