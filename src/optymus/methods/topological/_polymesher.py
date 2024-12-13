import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph, csr_matrix
from scipy.spatial import Voronoi
from tqdm import tqdm

plt.rcParams.update({"font.size": 12, "font.family": "serif"})


def polymesher(domain, n_elements, max_iter, initial_points=None, anim=False, plot=True):
    """PolyMesher

    Generate a polygon mesh using the polymesher algorithm.

    Parameters
    ----------
    domain : function
        The domain in which the mesh is generated.
    n_elements : int
        The number of elements in the mesh.
    max_iter : int
        The maximum number of iterations.
    initial_points : numpy.ndarray, optional
        Initial points for the mesh generation.
    anim : bool, optional
        If True, display an animation of the mesh generation.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "node": Node coordinates
        - "element": Element vertices
        - "boundary_supp": Boundary support conditions
        - "boundary_load": Boundary load conditions
        - "initial_points": Initial points used for mesh generation

    Examples
    --------
    >>> from optymus.benchmark import MbbDomain
    >>> from optymus.methods.topological import polymesher
    >>> mesh = polymesher(MbbDomain, n_elements=1000, max_iter=1000)

    Notes
    -----
    The PolyMesher algorithm generates a polygon mesh using a Voronoi diagram.

    References
    ----------
    [1] - Talischi, C., Paulino, G. H., Pereira, A., & Menezes, I. F. (2012). "PolyMesher: a general-purpose mesh generator for polygonal elements written in Matlab". Structural and Multidisciplinary Optimization, 45, 309-328.
    """
    if initial_points is None:
        initial_points = poly_random_point_set(n_elements, domain)

    n_elements = initial_points.shape[0]
    tolerance = 1e-6
    pointer = 0
    error = 1
    c = 1.5  # constant of proportionality used for calculation of 'alpha' should be greater than 1
    domain_bounding_box = domain.domain_bounding_box
    domain_fixed_points = np.array(domain.domain_fixed_points).reshape((-1, 2))
    area = (domain_bounding_box[1] - domain_bounding_box[0]) * (domain_bounding_box[3] - domain_bounding_box[2])
    initial_points_copy = initial_points.copy()
    time_acc = 0
    if anim and plot is True:
        fig, ax = plt.subplots(figsize=(8, 5))

    pbar = tqdm(total=max_iter, desc="Iterations", unit="step")
    while pointer < max_iter and error > tolerance:
        start_time = time.time()
        alpha = c * np.sqrt(area / n_elements)  # a distance value proportional to the width of an element
        initial_points = initial_points_copy.copy()
        reflected_points = poly_reflected_points(initial_points, n_elements, domain, alpha)
        initial_points, reflected_points = poly_fixed_points(initial_points, reflected_points, domain_fixed_points)

        vor = Voronoi(np.vstack((initial_points, reflected_points)))
        node = vor.vertices

        # Reordering regions based on points
        element = [vor.regions[reg] for reg in vor.point_region]

        initial_points_copy, areas = poly_centroids(element, node, n_elements)
        area = np.sum(np.abs(areas))
        error = (
            np.sqrt(np.sum((areas**2) * np.sum((initial_points_copy - initial_points) ** 2, axis=1)))
            * n_elements
            / (area**1.5)
        )  # error calculation
        pointer += 1
        end_time = time.time()
        time_acc += end_time - start_time

        pbar.update(1)
        pbar.set_postfix({"Error": error, "Iteration": pointer, "Time": time_acc})

        if anim and n_elements <= 2000 and plot is True:
            ax.clear()  # Clear the axis for redrawing
            plot_mesh(element, n_elements, node, pointer, error, anim=True, ax=ax)

    if anim and plot is True:
        plt.show()

    node, element = poly_unique_nodes(node_coordinates=node, element_vertices=element[:n_elements])
    node, element = poly_collapse_small_edges(node_coordinates=node, element_vertices=element, eps=0.1)
    node, element = poly_rearrange_nodes(node_coordinates=node, element_vertices=element, n_elements=n_elements)

    domain_boundary_conditions = domain.boundary_conditions(node)
    boundary_supp = domain_boundary_conditions[0]
    boundary_load = domain_boundary_conditions[1]

    if anim is False and plot is True:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_mesh(element=element, n_elements=n_elements, node=node, pointer=pointer, error=error, ax=ax)

    return {
        "node": node,
        "element": element,
        "boundary_supp": boundary_supp,
        "boundary_load": boundary_load,
        "initial_points": initial_points,
    }


def plot_mesh(element, n_elements, node, pointer, error, anim=False, ax=None):
    element = element[:n_elements]
    Node_set = set()
    for polygon in element:
        if -1 in polygon:  # Handle infinite regions
            continue
        vx = [node[i, 0] for i in polygon]
        vy = [node[i, 1] for i in polygon]
        Node_set.update(polygon)
        ax.fill(vx, vy, edgecolor="black", facecolor="white", alpha=0.8)
        # add a point at the centroid of the polygon
        ax.plot(np.mean(vx), np.mean(vy), color="black", marker="o", markersize=2)

    # Plot Voronoi vertices
    Node_set = node[list(Node_set)]
    ax.plot(Node_set[:, 0], Node_set[:, 1], color="navy", linestyle="None")
    ax.set_title(f"Iteration: {pointer}, Error: {error:.4f}")
    ax.set_axis_off()

    if anim:
        plt.pause(0.0000001)
    else:
        plt.show()


def poly_random_point_set(n_elements, domain):
    """
    Generate an initial random point set of size 'n_elements' for polygon mesh generation.

    Args:
        n_elements (int): The number of points to generate.
        domain (function): The domain in which points are generated.

    Returns:
        numpy.ndarray: A 2D array containing the generated points.
    """
    points = np.zeros((n_elements, 2))
    domain_bounding_box = domain.domain_bounding_box
    Ctr = 0

    while Ctr < n_elements:
        Y = np.random.rand(n_elements, 2)
        Y[:, 0] = (domain_bounding_box[1] - domain_bounding_box[0]) * Y[:, 0] + domain_bounding_box[0]
        Y[:, 1] = (domain_bounding_box[3] - domain_bounding_box[2]) * Y[:, 1] + domain_bounding_box[2]
        d = domain.signed_distance(Y)
        I = np.where(d[:, -1] < 0)[0]
        NumAdded = min(n_elements - Ctr, len(I))
        points[Ctr : Ctr + NumAdded, :] = Y[I[0:NumAdded], :]
        Ctr += NumAdded

    return points


def poly_fixed_points(original_points, reflected_points, domain_fixed_points):
    """
    Adjust points based on fixed points to maintain mesh quality.

    Args:
        original_points (numpy.ndarray): Original points.
        reflected_points (numpy.ndarray): Reflected points.
        domain_fixed_points (numpy.ndarray): Fixed points.

    Returns:
        tuple: A tuple containing adjusted points P and reflected_points.
    """
    stack_original_points = np.vstack((original_points, reflected_points))
    for i in range(domain_fixed_points.shape[0]):
        B, I = (
            np.sort(
                np.sqrt(
                    (stack_original_points[:, 0] - domain_fixed_points[i, 0]) ** 2
                    + (stack_original_points[:, 1] - domain_fixed_points[i, 1]) ** 2
                )
            ),
            np.argsort(
                np.sqrt(
                    (stack_original_points[:, 0] - domain_fixed_points[i, 0]) ** 2
                    + (stack_original_points[:, 1] - domain_fixed_points[i, 1]) ** 2
                )
            ),
        )
        for j in range(1, 4):
            n = stack_original_points[I[j], :] - domain_fixed_points[i, :]
            n = n / np.linalg.norm(n)
            stack_original_points[I[j], :] = stack_original_points[I[j], :] - n * (B[j] - B[0])

    new_original_points = stack_original_points[: original_points.shape[0], :]
    reflected_points = stack_original_points[new_original_points.shape[0] :, :]
    return new_original_points, reflected_points


def poly_reflected_points(original_points, n_elements, domain, alpha):
    """
    Reflect points at the boundary for mesh generation.

    Args:
        original_points (numpy.ndarray): Original points.
        n_elements (int): Number of elements.
        domain (function): The domain for boundary checks.
        alpha (float): The propotional distance value.

    Returns:
        numpy.ndarray: Reflected points.
    """
    eps = 1e-8  # Small positive number for numerical differentiation
    # A specified parameter (0<eta<1) to adjust for numerical errors (round-off and numerical differentiation)
    eta = 0.9
    d = domain.signed_distance(original_points)
    NBdrySegs = d.shape[1] - 1

    # The gradient of the distance function is computed by means of numerical differentiation
    n1 = ((domain.signed_distance(original_points + np.array([[eps, 0]] * n_elements))) - d) / eps
    n2 = ((domain.signed_distance(original_points + np.array([[0, eps]] * n_elements))) - d) / eps
    I = np.abs(d[:, 0:NBdrySegs]) < alpha
    P1 = np.broadcast_to(original_points[:, 0][:, np.newaxis], (original_points[:, 0].shape[0], NBdrySegs))
    P2 = np.broadcast_to(original_points[:, 1][:, np.newaxis], (original_points[:, 1].shape[0], NBdrySegs))

    P1 = P1[I] - 2 * n1[:, 0:NBdrySegs][I] * d[:, 0:NBdrySegs][I]
    P2 = P2[I] - 2 * n2[:, 0:NBdrySegs][I] * d[:, 0:NBdrySegs][I]
    reflected_points = np.vstack((P1, P2)).T
    d_reflected_points = domain.signed_distance(reflected_points)
    J = (np.abs(d_reflected_points[:, -1]) >= eta * np.abs(d[:, 0:NBdrySegs][I])) & (d_reflected_points[:, -1] > 0)

    reflected_points = np.unique(reflected_points[J, :], axis=0)
    return reflected_points


def poly_centroids(element_vertices, node, n_elements):
    """
    Compute centroids and areas for elements in the mesh.

    Args:
        element_vertices (list): List of element vertices.
        Node (numpy.ndarray): Node coordinates.
        n_elements (int): Number of elements to consider.

    Returns:
        tuple: A tuple containing centroids and areas.
    """
    centroids = []
    areas = []
    counter = 0
    for vertices in element_vertices:
        if counter >= n_elements:
            break
        if -1 in vertices:
            continue
        if len(vertices) >= 3:
            # Extract the vertex coordinates
            polygon_vertices = node[vertices]

            # Compute the area of the polygon using the shoelace formula
            vx = polygon_vertices[:, 0]
            vy = polygon_vertices[:, 1]
            vxs = np.roll(vx, 1)
            vys = np.roll(vy, 1)
            temp = vx * vys - vy * vxs
            area = 0.5 * np.sum(temp)
            areas.append(area)

            # Compute the centroid of the polygon
            centroid = 1 / (6 * area) * np.array([np.sum((vx + vxs) * temp), np.sum((vy + vys) * temp)])
            centroids.append(centroid)

            counter += 1

    return np.array(centroids), np.array(areas)


def poly_unique_nodes(node_coordinates, element_vertices):
    """
    Extract unique nodes and rebuild node and element lists.

    Args:
        node_coordinates (numpy.ndarray): Original node coordinates.
        element_vertices (list): List of element vertices.

    Returns:
        tuple: A tuple containing updated Node and Element.
    """
    unique_nodes = np.unique(np.concatenate(element_vertices))
    c_node = np.arange(len(node_coordinates))
    c_node[~np.in1d(c_node, unique_nodes)] = np.max(unique_nodes)
    node, element = poly_rebuild_node_mapping(node_coordinates, element_vertices, c_node)
    return node, element


def poly_collapse_small_edges(node_coordinates, element_vertices, eps):
    """
    Collapse small edges based on a specified tolerance.

    Args:
        node_coordinates (numpy.ndarray): Node coordinates.
        element_vertices (list): List of element vertices.
        eps (float): Tolerance for edge collapse.

    Returns:
        tuple: A tuple containing updated Node and Element.
    """
    while True:
        c_edge = []
        for ele in element_vertices:
            if len(ele) < 4:
                continue  # Cannot collapse triangles
            vx = node_coordinates[ele, 0]
            vy = node_coordinates[ele, 1]
            nv = len(vx)
            beta = np.arctan2(vy - np.sum(vy) / nv, vx - np.sum(vx) / nv)
            beta = np.mod(beta[np.roll(np.arange(len(beta)), shift=-1)] - beta, 2 * np.pi)
            beta_ideal = 2 * np.pi / len(ele)
            edge = np.column_stack((ele, np.roll(ele, shift=-1)))
            c_edge.extend(edge[beta < eps * beta_ideal, :])

        if len(c_edge) == 0:
            break

        c_edge = np.unique(np.sort(c_edge, axis=1), axis=0)
        c_node = np.arange(len(node_coordinates))
        for i in range(c_edge.shape[0]):
            c_node[c_edge[i, 1]] = c_node[c_edge[i, 0]]
        node_coordinates, element_vertices = poly_rebuild_node_mapping(node_coordinates, element_vertices, c_node)
    return node_coordinates, element_vertices


def poly_rearrange_nodes(node_coordinates, element_vertices, n_elements):  # noqa
    """
    Rearrange nodes to improve mesh quality using (RCM) algorithm.

    Args:
        node_coordinates (numpy.ndarray): Original node coordinates.
        element_vertices (list): List of element vertices.
        n_elements (int): Number of elements to consider.

    Returns:
        tuple: A tuple containing updated Node and Element.
    """
    node_coordinates_size = node_coordinates.shape[0]
    element_vertices_size = len(element_vertices)

    element_vertices_lenght = [len(e) for e in element_vertices]
    nn = np.sum(np.array(element_vertices_lenght) ** 2)

    i = np.zeros(nn, dtype=int)
    j = np.zeros(nn, dtype=int)
    s = np.ones(nn)
    index = 0

    for el in range(element_vertices_size):
        eNode = element_vertices[el]
        ElemSet = np.arange(index, index + element_vertices_lenght[el] ** 2)
        i[ElemSet] = np.kron(eNode, np.ones(element_vertices_lenght[el], dtype=int))
        j[ElemSet] = np.kron(eNode, np.ones(element_vertices_lenght[el], dtype=int))
        index += element_vertices_lenght[el] ** 2

    K = csr_matrix((s, (i, j)), shape=(node_coordinates_size, node_coordinates_size))
    p = csgraph.reverse_cuthill_mckee(K)

    cNode = np.arange(0, node_coordinates_size)
    cNode[p[:node_coordinates_size]] = np.arange(0, node_coordinates_size)

    node, element = poly_rebuild_node_mapping(node_coordinates, element_vertices, cNode)

    return node, element


def poly_rebuild_node_mapping(node_coordinates, element_vertices, c_node):
    """
    Rebuild node and element lists based on node mapping.

    Args:
        node_coordinates (numpy.ndarray): Original node coordinates.
        element_vertices (list): List of element vertices.
        c_node (numpy.ndarray): Node mapping.

    Returns:
        tuple: A tuple containing updated Node and Element.
    """
    element = [None] * len(element_vertices)
    _, ix, jx = np.unique(c_node, return_index=True, return_inverse=True)

    if not np.array_equal(jx.shape, c_node.shape):
        jx = jx.T

    if node_coordinates.shape[0] > len(ix):
        ix[-1] = max(c_node)

    node = node_coordinates[ix]

    for i, j in enumerate(element_vertices):
        element[i] = np.unique(jx[j])
        vx = node[element[i], 0]
        vy = node[element[i], 1]
        nv = len(vx)
        angles = np.arctan2(vy - np.sum(vy) / nv, vx - np.sum(vx) / nv)
        iix = np.argsort(angles)
        element[i] = element[i][iix]

    return node, element


def mesh_assessment(Node, Element, domain_area=0, verbose=True):
    """
    Assesses the quality of a mesh based on element aspect ratio and element area.

    This function calculates the following mesh quality metrics:
    * Maximum aspect ratio (AR) of all elements
    * Average aspect ratio of all elements
    * Average edge length across all elements
    * Range of element areas (minimum and maximum)
    * Standard deviation of element areas
    * Total area error between domain area and total element areas

    Args:
    Node (numpy.ndarray): Node coordinates.
    Element (list): List of element vertices.
    domain_area (float): Area of the domain (optional).
    verbose (boolean): Print mesh quality metrics (optional).

    Returns:
    dict: A dictionary containing the calculated mesh quality metrics.
    - "Max. Mesh AR": Maximum aspect ratio of all elements.
    - "Average Mesh AR": Average aspect ratio of all elements.
    - "Avg. Length": Average edge length across all elements.
    - "Range of Areas": Tuple containing the minimum and maximum element areas.
    - "Standard Deviation of Areas": Standard deviation of element areas.
    - "Total Area Error (%)": Total area error between domain area and total element areas

    Prints:
    The calculated mesh quality metrics to the console for immediate feedback.
    """
    assessment = {}
    mesh_ar = []
    all_lengths = []
    areas = []

    for elem in Element:
        elem_nodes = Node[elem]
        elem_nodes_rolled = np.roll(elem_nodes, 1, axis=0)

        # Calculate edge lengths
        edge_vectors = elem_nodes - elem_nodes_rolled
        lengths = np.sqrt(np.sum(edge_vectors**2, axis=1))

        mesh_ar.append(np.max(lengths) / np.min(lengths))
        all_lengths.extend(lengths)

        # Calculate element area
        vx, vy = elem_nodes[:, 0], elem_nodes[:, 1]
        vxs, vys = np.roll(vx, -1), np.roll(vy, -1)
        area = 0.5 * np.abs(np.sum(vx * vys - vy * vxs))
        areas.append(area)

    assessment["Max. Mesh AR"] = np.max(mesh_ar)
    assessment["Average Mesh AR"] = np.mean(mesh_ar)
    assessment["Avg. Length"] = np.mean(all_lengths)
    assessment["Range of Areas"] = (np.min(areas), np.max(areas))
    assessment["Standard Deviation of Areas"] = np.std(areas)

    if domain_area:
        total_area = np.sum(areas)
        area_error = 100 * (total_area - domain_area) / domain_area
        assessment["Total Area Error (%)"] = area_error

    if verbose:
        for crit, val in assessment.items():
            print(f"{crit}: {val}")  # noqa

    return assessment
