"""
Topology Optimization using Optymus Optimizers.

This module provides high-level functions for running topology optimization
using the optymus optimization methods.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ._compliance import ComplianceObjective


def topology_optimization(mesh, method="bfgs", volume_fraction=0.5,
                          max_iter=100, E0=1.0, nu=0.3, penal=3.0,
                          filter_radius=None, learning_rate=0.01,
                          penalty_weight=1e6, verbose=True, **kwargs):
    """
    Perform topology optimization using optymus optimizers.

    Parameters
    ----------
    mesh : dict
        Output from polymesher containing node, element, boundary_supp, boundary_load
    method : str
        Optimization method. Options include:
        - Gradient-based: 'adam', 'bfgs', 'lbfgs', 'steepest_descent', 'conjugate_gradient'
        - Adaptive: 'rmsprop', 'adagrad', 'adamax', 'yogi'
        - Evolutionary: 'cmaes', 'particle_swarm', 'differential_evolution'
        - Other: 'simulated_annealing', 'powell'
    volume_fraction : float
        Target volume fraction (0 < vf < 1)
    max_iter : int
        Maximum number of iterations
    E0 : float
        Young's modulus of solid material
    nu : float
        Poisson's ratio
    penal : float
        SIMP penalization power (typically 3)
    filter_radius : float, optional
        Sensitivity filter radius for eliminating checkerboard patterns.
        If None, no filtering is applied.
    learning_rate : float
        Learning rate for gradient-based methods
    penalty_weight : float
        Penalty weight for volume constraint (default 1e6).
        Higher values enforce the constraint more strictly.
    verbose : bool
        Show progress bar
    **kwargs
        Additional arguments passed to the optimizer

    Returns
    -------
    dict
        Optimization results containing:
        - xopt: optimal densities
        - fmin: minimum compliance
        - densities: same as xopt (convenience)
        - num_iter: number of iterations
        - path: optimization path
        - compliance_history: compliance values per iteration
        - volume_history: volume fraction per iteration
        - method_name: name of method used
    """
    from optymus import Optimizer

    # Create compliance objective
    compliance_obj = ComplianceObjective(
        mesh,
        E0=E0,
        nu=nu,
        penal=penal,
        volume_fraction=volume_fraction,
        filter_radius=filter_radius
    )

    n_elements = compliance_obj.n_elements

    # Initial densities (uniform at volume fraction)
    x0 = jnp.ones(n_elements) * volume_fraction

    # Create JAX-differentiable objective with custom VJP
    # This allows JAX's autodiff to work with the NumPy/SciPy-based FEM solver
    @jax.custom_vjp
    def f_obj(x):
        """Compliance objective - differentiable via custom_vjp."""
        x_np = np.asarray(x)
        return jnp.array(compliance_obj(x_np), dtype=jnp.float64)

    def f_obj_fwd(x):
        """Forward pass: evaluate compliance."""
        val = f_obj(x)
        return val, x  # Return value and residuals for backward

    def f_obj_bwd(res, g):
        """Backward pass: analytical sensitivity."""
        x = res
        x_np = np.asarray(x)
        grad = compliance_obj.gradient(x_np)
        # Apply sensitivity filter if configured
        grad = compliance_obj.filter_sensitivities(x_np, grad)
        return (jnp.array(grad, dtype=jnp.float64) * g,)

    f_obj.defvjp(f_obj_fwd, f_obj_bwd)

    # Create JAX-differentiable volume constraint with custom VJP
    # Scale constraint by sqrt(penalty_weight) so penalty method (which squares)
    # gives effective penalty of penalty_weight * constraint^2
    penalty_scale = jnp.sqrt(penalty_weight)

    @jax.custom_vjp
    def f_cons(x):
        """Volume constraint - scaled by penalty weight for effective enforcement."""
        x_np = np.asarray(x)
        return jnp.array(compliance_obj.volume_constraint(x_np) * penalty_scale, dtype=jnp.float64)

    def f_cons_fwd(x):
        """Forward pass: evaluate scaled volume constraint."""
        val = f_cons(x)
        return val, x

    def f_cons_bwd(res, g):
        """Backward pass: scaled volume constraint gradient."""
        x = res
        x_np = np.asarray(x)
        grad = compliance_obj.volume_constraint_gradient(x_np)
        return (jnp.array(grad * penalty_scale, dtype=jnp.float64) * g,)

    f_cons.defvjp(f_cons_fwd, f_cons_bwd)

    # Run optimization
    opt = Optimizer(
        f_obj=f_obj,
        f_cons=[f_cons],
        x0=x0,
        method=method,
        bounds=(0.001, 1.0),
        max_iter=max_iter,
        learning_rate=learning_rate,
        verbose=verbose,
        **kwargs
    )

    results = opt.get_results()

    # Add topology-specific results
    results["densities"] = np.array(results["xopt"])
    results["compliance_obj"] = compliance_obj

    # Compute final volume fraction
    final_volume = np.sum(results["densities"] * compliance_obj.element_volumes)
    results["final_volume_fraction"] = final_volume / compliance_obj.total_volume

    return results


def optimality_criteria(mesh, volume_fraction=0.5, max_iter=100,
                        E0=1.0, nu=0.3, penal=3.0, tol=1e-3,
                        move=0.2, filter_radius=None, verbose=True):
    """
    Topology optimization using the Optimality Criteria (OC) method.

    This is a specialized method for compliance minimization that is
    typically faster and more stable than general-purpose optimizers
    for this specific problem.

    Parameters
    ----------
    mesh : dict
        Output from polymesher
    volume_fraction : float
        Target volume fraction
    max_iter : int
        Maximum iterations
    E0 : float
        Young's modulus
    nu : float
        Poisson's ratio
    penal : float
        SIMP penalization power
    tol : float
        Convergence tolerance on density change
    move : float
        Move limit for density update
    filter_radius : float, optional
        Density filter radius for eliminating checkerboard patterns.
        If None, no filtering is applied. Recommended value:
        1.5 * sqrt(total_area / n_elements)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Optimization results
    """
    compliance_obj = ComplianceObjective(
        mesh,
        E0=E0,
        nu=nu,
        penal=penal,
        volume_fraction=volume_fraction,
        filter_radius=filter_radius
    )

    n_elements = compliance_obj.n_elements
    element_volumes = compliance_obj.element_volumes
    total_volume = compliance_obj.total_volume
    target_volume = volume_fraction * total_volume

    # Initialize densities
    x = np.ones(n_elements) * volume_fraction
    x_old = x.copy()

    compliance_history = []
    volume_history = []

    if verbose:
        print(f"{'Iter':>5} {'Compliance':>12} {'Volume':>10} {'Change':>10}")
        print("-" * 40)

    for iteration in range(max_iter):
        # Compute compliance and sensitivity
        c = compliance_obj(x)
        dc = compliance_obj.gradient(x)

        # Apply sensitivity filter if filter_radius is set
        dc = compliance_obj.filter_sensitivities(x, dc)

        compliance_history.append(c)
        volume_history.append(np.sum(x * element_volumes) / total_volume)

        # Optimality criteria update
        # Find Lagrange multiplier using bisection
        l1, l2 = 1e-9, 1e9

        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
            lmid = 0.5 * (l1 + l2)

            # OC update formula: x_new = x * (-dc / lmid)^eta
            # For compliance, eta = 0.5 and dc is negative
            # So B = -dc / lmid should be positive
            B = np.abs(dc) / (lmid + 1e-12)
            B = np.maximum(B, 1e-12)

            x_new = x * np.power(B, 0.5)

            # Apply move limits
            x_new = np.maximum(x - move, np.minimum(x + move, x_new))
            x_new = np.clip(x_new, 0.001, 1.0)

            # Check volume constraint
            if np.sum(x_new * element_volumes) > target_volume:
                l1 = lmid
            else:
                l2 = lmid

        x = x_new

        # Check convergence
        change = np.max(np.abs(x - x_old))
        x_old = x.copy()

        if verbose:
            print(f"{iteration:5d} {c:12.4f} {volume_history[-1]:10.4f} {change:10.6f}")

        if change < tol and iteration > 10:
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations")
            break

    return {
        "xopt": x,
        "fmin": compliance_history[-1],
        "densities": x,
        "num_iter": iteration + 1,
        "compliance_history": np.array(compliance_history),
        "volume_history": np.array(volume_history),
        "method_name": "Optimality Criteria (OC)"
    }


def plot_topology(mesh, densities, ax=None, cmap="gray_r", title=None,
                  show_colorbar=True, threshold=None):
    """
    Visualize topology optimization result.

    Parameters
    ----------
    mesh : dict
        Mesh from polymesher
    densities : ndarray
        Element densities
    ax : matplotlib axis, optional
        Axis to plot on
    cmap : str
        Colormap (default: gray_r for black=solid, white=void)
    title : str, optional
        Plot title
    show_colorbar : bool
        Whether to show colorbar
    threshold : float, optional
        If provided, threshold densities (0/1) for crisp visualization

    Returns
    -------
    ax : matplotlib axis
        The axis with the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    nodes = mesh["node"]
    elements = mesh["element"]

    # Apply threshold if specified
    if threshold is not None:
        plot_densities = np.where(densities >= threshold, 1.0, 0.0)
    else:
        plot_densities = densities

    patches = []
    for elem_nodes in elements:
        polygon = Polygon(nodes[elem_nodes], closed=True)
        patches.append(polygon)

    collection = PatchCollection(patches, cmap=cmap, edgecolor="none")
    collection.set_array(plot_densities)
    collection.set_clim(0, 1)

    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(collection, ax=ax, label=r"Density $\rho$", shrink=0.8)

    ax.axis("off")

    return ax


def plot_convergence(compliance_history, volume_history=None, ax=None):
    """
    Plot optimization convergence.

    Parameters
    ----------
    compliance_history : ndarray
        Compliance values per iteration
    volume_history : ndarray, optional
        Volume fraction per iteration
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    ax : matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    iterations = np.arange(len(compliance_history))

    ax.semilogy(iterations, compliance_history, 'k-', linewidth=1.5,
                label='Compliance')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Compliance')
    ax.grid(True, alpha=0.3)

    if volume_history is not None:
        ax2 = ax.twinx()
        ax2.plot(iterations, volume_history, 'b--', linewidth=1.0,
                 label='Volume fraction')
        ax2.set_ylabel('Volume fraction', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(0, 1)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax.legend(loc='upper right')

    ax.set_title('Optimization Convergence')

    return ax
