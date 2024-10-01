import numpy as np
import matplotlib.pyplot as plt

from optymus.benchmark.utils._domain_functions import dCircle, dDiff, dIntersect, dLine, dRectangle, dUnion

class TopologicalDomain:
    """
    Represents a mathematically defined domain for polygonal mesh.

    This class defines a mathematically defined domain based on its name, bounding box,
    signed distance function (signed_distance_function), boundary conditions (domain_boundary_conditions), and fixed points (domain_fixed_points).

    Attributes:
        name (str): The name of the mathematically defined domain.
        domain_bounding_box (list): The bounding box of the domain, defined as [xmin, xmax, ymin, ymax].
        signed_distance_function (callable): The signed distance function that provides the distance values.
        domain_boundary_conditions (callable, optional): The function for setting boundary conditions. Default is None.
        domain_fixed_points (list, optional): List of fixed points within the domain. Default is an empty list.

    Methods:
        signed_distance(P):
            Computes the distance value for a point P using the signed distance function.

        boundary_conditions(Node):
            Determines the boundary conditions for a given node within the domain.

        Plot(n=1000):
            Plots the domain based on the signed distance function.

        compute_area(n=1_000_000):
            Calculates the approximate area of the domain using the Monte Carlo method
    """

    def __init__(self, name, domain_bounding_box, signed_distance_function, domain_boundary_conditions=None, domain_fixed_points=[]):
        """
        Initializes a Domain object with the provided attributes.

        Args:
            name (str): The name of the computational domain.
            domain_bounding_box (list): The bounding box of the domain, defined as [xmin, xmax, ymin, ymax].
            signed_distance_function (callable): The signed distance function that provides the distance values.
            domain_boundary_conditions (callable, optional): The function for setting boundary conditions. Default is None.
            domain_fixed_points (list, optional): List of fixed points within the domain. Default is an empty list.
        """
        self.name = name
        self.domain_bounding_box = domain_bounding_box
        self.domain_fixed_points = domain_fixed_points
        self.signed_distance_function = signed_distance_function
        self.domain_boundary_conditions = domain_boundary_conditions

    def signed_distance(self, P):
        """
        Computes the distance value for a point P using the signed distance function.

        Args:
            P (numpy.ndarray): A point or an array of points for which the distance is to be calculated.

        Returns:
            numpy.ndarray: An array of distance values corresponding to the input points.
        """
        return self.signed_distance_function(P)

    def boundary_conditions(self, Node):
        """
        Determines the boundary conditions for a given node within the domain.

        Args:
            Node (tuple): The coordinates of a node (x, y).

        Returns:
            list: A list containing boundary conditions for the node. It may include None values.
        """
        if self.domain_boundary_conditions is None:
            return [None, None]
        return self.domain_boundary_conditions(Node, self.domain_bounding_box)

    def plot(self, n=1000):
        """
        Plots the domain based on the signed distance function.

        Args:
            n (int, optional): The number of points for plotting. Default is 1000.

        Displays:
            A plot of the domain based on the signed distance function.
        """
        x, y = np.meshgrid(
            np.linspace(self.domain_bounding_box[0], self.domain_bounding_box[1], n),
            np.linspace(self.domain_bounding_box[2], self.domain_bounding_box[3], n),
        )
        points = np.hstack([x.reshape((-1, 1)), y.reshape((-1, 1))])
        signed_distance_function = self.signed_distance_function(points)[:, -1]

        inner = np.where(signed_distance_function <= 0, 1, 0)

        _, ax = plt.subplots(figsize=(8, 6))
        _ = ax.imshow(
            inner.reshape((n, n)),
            extent=(self.domain_bounding_box[0], self.domain_bounding_box[1], self.domain_bounding_box[2], self.domain_bounding_box[3]),
            origin="lower",
            cmap="Purples",
            alpha=0.8,
        )
        ax.contour(x, y, signed_distance_function.reshape((n, n)), levels=[0], colors="gold", linewidths=2)
        ax.set_xlabel("X", fontweight="bold")
        ax.set_ylabel("Y", fontweight="bold")
        ax.set_title("Domain Visualization", fontweight="bold", fontsize=16)
        ax.set_aspect("equal")

        plt.show()

    def compute_area(self, n=1_000_000):
        """
        Calculates the approximate area of the domain using the Monte Carlo method.

        Args:
            n (int, optional): The number of random points to use. Default is 1,000,000.

        Returns:
            float: The calculated approximate area of the domain.
        """
        xmin, xmax, ymin, ymax = self.domain_bounding_box
        total_area = (xmax - xmin) * (ymax - ymin)

        # Generate random points within the bounding box
        points = np.random.uniform(low=[xmin, ymin], high=[xmax, ymax], size=(n, 2))

        signed_distance_function_values = self.signed_distance_function(points)[:, -1] # Compute signed distance values

        points_inside = np.sum(signed_distance_function_values <= 0) # points inside the domain
        area_ratio = points_inside / n # ratio of points inside the domain
        approximate_area = area_ratio * total_area # approximate area of the domain

        return approximate_area


def _cookSDF(P):
    d1 = dLine(P, 0., 44., 0., 0.)
    d2 = dLine(P, 0., 0., 48., 44.)
    d3 = dLine(P, 48., 44., 48., 60.)
    d4 = dLine(P, 48., 60., 0., 44.)
    Dist = dIntersect(d4, dIntersect(d3, dIntersect(d2, d1)))
    return Dist

def _cookBC(Node, BdBox):
    eps = 0.1 * np.sqrt((BdBox[1] - BdBox[0]) *
                        (BdBox[3] - BdBox[2]) / Node.shape[0])
    LeftsideNodes = np.where(Node[:, 0] < eps)[0]
    Supp = np.ones((LeftsideNodes.shape[0], 3), dtype=int)
    Supp[:, 0] = LeftsideNodes

    RightsideNodes = np.where(Node[:, 0] > 48-eps)[0]
    Load = np.zeros((RightsideNodes.shape[0], 3), dtype=int)
    Load[:, 0] = RightsideNodes
    Load[:, 2] = 20
    return [Supp, Load]

CookDomain = TopologicalDomain("Cook Domain", [0, 48, 0, 60], _cookSDF, _cookBC)

def _SuspensionSDF(P):
    d1 = dRectangle(P, 0, 18.885, 0, 14.56)
    d2 = dLine(P, 18.885, 1.3030, 4, 0)
    d3 = dLine(P, 3.92, 14.56, 6.1699, 6.88)
    d4 = dLine(P, 9.8651, 4.0023, 18.885, 3.70)
    d5 = dLine(P, 4, 0, 0, 4)
    d13 = dLine(P, 0, 14, 3.92, 14.56)
    d14 = dCircle(P, 10, 8, 4)
    d15 = dLine(P, 9.8651, 4.0023, 6.1699, 6.88)
    d = dDiff(dDiff(dDiff(dDiff(d1, d2), d5), d13),
                dUnion(dDiff(dIntersect(d3, d4), d15), d14))
    d6 = dCircle(P, 2, 2, 2)
    d7 = dCircle(P, 4, 2, 2)
    d8 = dCircle(P, 2, 4, 2)
    d = dUnion(d, dUnion(d6, dUnion(d7, d8)))
    d9 = dCircle(P, 2, 14, 2)
    d10 = dCircle(P, 2, 16, 2)
    d11 = dCircle(P, 18.885, 2.5, 1.2)
    d12 = dCircle(P, 20, 2.5, 1.2)
    Dist = dUnion(d, dUnion(d9, dUnion(d10, dUnion(d11, d12))))
    return Dist

def _SuspensionBC(Node, BdBox):
    CornerCircle = np.sqrt(
        (Node[:, 0] - 2.0) ** 2 + (Node[:, 1] - 2.0) ** 2)
    CornerCircle = np.argsort(CornerCircle)
    UpperCircle = np.sqrt((Node[:, 0] - 2.0) **
                            2 + (Node[:, 1] - 16.0) ** 2)
    UpperCircle = np.argsort(UpperCircle)
    Supp = np.ones((2, 3), dtype=int)
    Supp[0, :] = [CornerCircle[0], 1, 1]
    Supp[1, :] = [UpperCircle[0], 1, 0]
    RightCircle = np.sqrt((Node[:, 0] - 20.0) **
                            2 + (Node[:, 1] - 2.5) ** 2)
    RightCircle = np.argsort(RightCircle)
    Load = np.ones((1, 3))
    Load[0, :] = np.array([RightCircle[0], -8, -1]).reshape((-1, 3))
    x = [Supp, Load]
    return x

SuspensionDomain = TopologicalDomain("Suspension Domain", [-2, 24, -2, 24], _SuspensionSDF, _SuspensionBC)

def _MichellSDF(P, BdBox = [0, 5, -2, 2]):
    d1 = dRectangle(P, BdBox[0], BdBox[1], BdBox[2], BdBox[3])
    d2 = dCircle(P, 0, 0, BdBox[3] / 2)
    Dist = dDiff(d1, d2)
    return Dist

def _MichellBC(Node, BdBox):
    eps = 0.1 * ((BdBox[1] - BdBox[0]) *
                    (BdBox[3] - BdBox[2]) / Node.shape[0]) ** 0.5
    CircleNodes = [i for i, (x, y) in enumerate(
        Node) if abs((x ** 2 + y ** 2) ** 0.5 - 1.0) < eps]
    Supp = np.array([[node, 1, 1]
                    for node in CircleNodes]).reshape((-1, 3))
    MidRightFace = [
        ((x - BdBox[1]) ** 2 + (y - (BdBox[2] + BdBox[3]) / 2) ** 2) for x, y in Node]
    MidRightFace = [i for i, _ in sorted(
        enumerate(MidRightFace), key=lambda x: x[1])]
    Load = np.array([MidRightFace[0], 0, -1]).reshape((-1, 3))
    x = [Supp, Load]
    return x

MichellDomain = TopologicalDomain("Michell Domain", [0, 5, -2, 2], _MichellSDF, _MichellBC)

def _WrenchSDF(P):
    d1 = dLine(P, 0, 0.3, 0, -0.3)
    d2 = dLine(P, 0, -0.3, 2, -0.5)
    d3 = dLine(P, 2, -0.5, 2, 0.5)
    d4 = dLine(P, 2, 0.5, 0, 0.3)
    d5 = dCircle(P, 0, 0, 0.3)
    d6 = dCircle(P, 2, 0, 0.5)
    douter = dUnion(d6, dUnion(d5,
                                dIntersect(d4, dIntersect(d3, dIntersect(d2, d1)))))
    d7 = dCircle(P, 0, 0, 0.175)
    d8 = dCircle(P, 2, 0, 0.3)
    din = dUnion(d8, d7)
    Dist = dDiff(douter, din)
    return Dist

def _WrenchBC(Node, BdBox):
    eps = 0.1 * np.sqrt((BdBox[1] - BdBox[0]) *
                        (BdBox[3] - BdBox[2]) / Node.shape[0])
    RightCircleNodes = np.where(
        np.abs(np.sqrt((Node[:, 0] - 2) ** 2 + Node[:, 1] ** 2) - 0.3) < eps)[0]
    Supp = np.ones((RightCircleNodes.shape[0], 3), dtype=int)
    Supp[:, 0] = RightCircleNodes
    LeftHalfCircleNodes = np.where(np.abs(np.maximum(
        np.sqrt(Node[:, 0] ** 2 + Node[:, 1] ** 2) - 0.175, Node[:, 1])) < eps)[0]
    Load = -0.1 * np.ones((LeftHalfCircleNodes.shape[0], 3))
    Load[:, 0] = LeftHalfCircleNodes
    Load[:, 1] = 0
    x = [Supp, np.array(Load)]
    return x

WrenchDomain = TopologicalDomain("Wrench Domain", [-0.3, 2.5, -0.5, 0.5], _WrenchSDF, _WrenchBC)

def _HornSDF(P):
    d1 = dCircle(P, 0, 0, 1)
    d2 = dCircle(P, -0.4, 0, 0.55)
    d3 = dLine(P, 0, 0, 1, 0)
    Dist = dIntersect(d3, dDiff(d1, d2))
    return Dist

HornDomain = TopologicalDomain("Horn Domain", [-1, 1, 0, 1], _HornSDF)

def _MbbSDF(P):
    Dist = dRectangle(P, 0, 3, 0, 1)
    return Dist

def _MbbBC(Node, BdBox):
    eps = 0.1 * np.sqrt((BdBox[1] - BdBox[0]) *
                        (BdBox[3] - BdBox[2]) / Node.shape[0])
    LeftEdgeNodes = np.where(np.abs(Node[:, 0] - 0) < eps)[0]
    LeftUpperNode = np.where(
        np.logical_and(
            np.abs(Node[:, 0] - 0
                    ) < eps, np.abs(Node[:, 1] - 1) < eps
        )
    )[0]
    RightBottomNode = np.where(
        np.logical_and(
            np.abs(Node[:, 0] - 3
                    ) < eps, np.abs(Node[:, 1] - 0) < eps
        )
    )[0]
    FixedNodes = np.concatenate((LeftEdgeNodes, RightBottomNode))
    Supp = np.zeros((len(FixedNodes), 3), dtype=int)
    Supp[:, 0] = FixedNodes
    Supp[:-1, 1] = 1
    Supp[-1, 2] = 1
    Load = np.zeros((1, 3))
    Load[0, 0], Load[0, 1], Load[0, 2] = LeftUpperNode[0], 0, -0.5
    x = [Supp, Load]
    return x

MbbDomain = TopologicalDomain("Mbb Domain", [-0.5, 3.5, -0.5, 1.5], _MbbSDF, _MbbBC)