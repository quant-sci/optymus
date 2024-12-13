import matplotlib.pyplot as plt
import numpy as np

from optymus.benchmark.utils._domain_functions import dcircle, ddiff, dintersect, dline, drectangle, dunion

plt.rcParams.update({"font.size": 12, "font.family": "serif"})


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

    def __init__(
        self,
        name,
        domain_bounding_box,
        signed_distance_function,
        domain_boundary_conditions=None,
        domain_fixed_points=[],  # noqa
    ):
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
            extent=(
                self.domain_bounding_box[0],
                self.domain_bounding_box[1],
                self.domain_bounding_box[2],
                self.domain_bounding_box[3],
            ),
            origin="lower",
            cmap="Blues",
            alpha=0.8,
        )
        ax.grid(linewidth=0.5, linestyle="--", color="gray", alpha=0.5)
        ax.contour(x, y, signed_distance_function.reshape((n, n)), levels=[0], colors="gray", linewidths=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Domain Visualization", fontsize=14)
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

        signed_distance_function_values = self.signed_distance_function(points)[:, -1]  # Compute signed distance values

        points_inside = np.sum(signed_distance_function_values <= 0)  # points inside the domain
        area_ratio = points_inside / n  # ratio of points inside the domain
        approximate_area = area_ratio * total_area  # approximate area of the domain

        return approximate_area


def _cooksdf(P):
    d1 = dline(P, 0.0, 44.0, 0.0, 0.0)
    d2 = dline(P, 0.0, 0.0, 48.0, 44.0)
    d3 = dline(P, 48.0, 44.0, 48.0, 60.0)
    d4 = dline(P, 48.0, 60.0, 0.0, 44.0)
    Dist = dintersect(d4, dintersect(d3, dintersect(d2, d1)))
    return Dist


def _cookbc(Node, bdbox):
    eps = 0.1 * np.sqrt((bdbox[1] - bdbox[0]) * (bdbox[3] - bdbox[2]) / Node.shape[0])
    LeftsideNodes = np.where(Node[:, 0] < eps)[0]
    Supp = np.ones((LeftsideNodes.shape[0], 3), dtype=int)
    Supp[:, 0] = LeftsideNodes

    RightsideNodes = np.where(Node[:, 0] > 48 - eps)[0]
    Load = np.zeros((RightsideNodes.shape[0], 3), dtype=int)
    Load[:, 0] = RightsideNodes
    Load[:, 2] = 20
    return [Supp, Load]


CookDomain = TopologicalDomain("Cook Domain", [0, 48, 0, 60], _cooksdf, _cookbc)


def _suspensionsdf(P):
    d1 = drectangle(P, 0, 18.885, 0, 14.56)
    d2 = dline(P, 18.885, 1.3030, 4, 0)
    d3 = dline(P, 3.92, 14.56, 6.1699, 6.88)
    d4 = dline(P, 9.8651, 4.0023, 18.885, 3.70)
    d5 = dline(P, 4, 0, 0, 4)
    d13 = dline(P, 0, 14, 3.92, 14.56)
    d14 = dcircle(P, 10, 8, 4)
    d15 = dline(P, 9.8651, 4.0023, 6.1699, 6.88)
    d = ddiff(ddiff(ddiff(ddiff(d1, d2), d5), d13), dunion(ddiff(dintersect(d3, d4), d15), d14))
    d6 = dcircle(P, 2, 2, 2)
    d7 = dcircle(P, 4, 2, 2)
    d8 = dcircle(P, 2, 4, 2)
    d = dunion(d, dunion(d6, dunion(d7, d8)))
    d9 = dcircle(P, 2, 14, 2)
    d10 = dcircle(P, 2, 16, 2)
    d11 = dcircle(P, 18.885, 2.5, 1.2)
    d12 = dcircle(P, 20, 2.5, 1.2)
    Dist = dunion(d, dunion(d9, dunion(d10, dunion(d11, d12))))
    return Dist


def _suspensionbc(Node, bdbox):
    CornerCircle = np.sqrt((Node[:, 0] - 2.0) ** 2 + (Node[:, 1] - 2.0) ** 2)
    CornerCircle = np.argsort(CornerCircle)
    UpperCircle = np.sqrt((Node[:, 0] - 2.0) ** 2 + (Node[:, 1] - 16.0) ** 2)
    UpperCircle = np.argsort(UpperCircle)
    Supp = np.ones((2, 3), dtype=int)
    Supp[0, :] = [CornerCircle[0], 1, 1]
    Supp[1, :] = [UpperCircle[0], 1, 0]
    RightCircle = np.sqrt((Node[:, 0] - 20.0) ** 2 + (Node[:, 1] - 2.5) ** 2)
    RightCircle = np.argsort(RightCircle)
    Load = np.ones((1, 3))
    Load[0, :] = np.array([RightCircle[0], -8, -1]).reshape((-1, 3))
    x = [Supp, Load]
    return x


SuspensionDomain = TopologicalDomain("Suspension Domain", [-2, 24, -2, 24], _suspensionsdf, _suspensionbc)


def _michellsdf(P, bdbox=[0, 5, -2, 2]):  # noqa
    d1 = drectangle(P, bdbox[0], bdbox[1], bdbox[2], bdbox[3])
    d2 = dcircle(P, 0, 0, bdbox[3] / 2)
    Dist = ddiff(d1, d2)
    return Dist


def _michellbc(Node, bdbox):
    eps = 0.1 * ((bdbox[1] - bdbox[0]) * (bdbox[3] - bdbox[2]) / Node.shape[0]) ** 0.5
    CircleNodes = [i for i, (x, y) in enumerate(Node) if abs((x**2 + y**2) ** 0.5 - 1.0) < eps]
    Supp = np.array([[node, 1, 1] for node in CircleNodes]).reshape((-1, 3))
    MidRightFace = [((x - bdbox[1]) ** 2 + (y - (bdbox[2] + bdbox[3]) / 2) ** 2) for x, y in Node]
    MidRightFace = [i for i, _ in sorted(enumerate(MidRightFace), key=lambda x: x[1])]
    Load = np.array([MidRightFace[0], 0, -1]).reshape((-1, 3))
    x = [Supp, Load]
    return x


MichellDomain = TopologicalDomain("Michell Domain", [0, 5, -2, 2], _michellsdf, _michellbc)


def _wrenchsdf(P):
    d1 = dline(P, 0, 0.3, 0, -0.3)
    d2 = dline(P, 0, -0.3, 2, -0.5)
    d3 = dline(P, 2, -0.5, 2, 0.5)
    d4 = dline(P, 2, 0.5, 0, 0.3)
    d5 = dcircle(P, 0, 0, 0.3)
    d6 = dcircle(P, 2, 0, 0.5)
    douter = dunion(d6, dunion(d5, dintersect(d4, dintersect(d3, dintersect(d2, d1)))))
    d7 = dcircle(P, 0, 0, 0.175)
    d8 = dcircle(P, 2, 0, 0.3)
    din = dunion(d8, d7)
    Dist = ddiff(douter, din)
    return Dist


def _wrenchbc(Node, bdbox):
    eps = 0.1 * np.sqrt((bdbox[1] - bdbox[0]) * (bdbox[3] - bdbox[2]) / Node.shape[0])
    RightCircleNodes = np.where(np.abs(np.sqrt((Node[:, 0] - 2) ** 2 + Node[:, 1] ** 2) - 0.3) < eps)[0]
    Supp = np.ones((RightCircleNodes.shape[0], 3), dtype=int)
    Supp[:, 0] = RightCircleNodes
    LeftHalfCircleNodes = np.where(
        np.abs(np.maximum(np.sqrt(Node[:, 0] ** 2 + Node[:, 1] ** 2) - 0.175, Node[:, 1])) < eps
    )[0]
    Load = -0.1 * np.ones((LeftHalfCircleNodes.shape[0], 3))
    Load[:, 0] = LeftHalfCircleNodes
    Load[:, 1] = 0
    x = [Supp, np.array(Load)]
    return x


WrenchDomain = TopologicalDomain("Wrench Domain", [-0.3, 2.5, -0.5, 0.5], _wrenchsdf, _wrenchbc)


def _hornsdf(P):
    d1 = dcircle(P, 0, 0, 1)
    d2 = dcircle(P, -0.4, 0, 0.55)
    d3 = dline(P, 0, 0, 1, 0)
    Dist = dintersect(d3, ddiff(d1, d2))
    return Dist


HornDomain = TopologicalDomain("Horn Domain", [-1, 1, 0, 1], _hornsdf)


def _mbbsdf(P):
    Dist = drectangle(P, 0, 3, 0, 1)
    return Dist


def _mbbbc(Node, bdbox):
    eps = 0.1 * np.sqrt((bdbox[1] - bdbox[0]) * (bdbox[3] - bdbox[2]) / Node.shape[0])
    LeftEdgeNodes = np.where(np.abs(Node[:, 0] - 0) < eps)[0]
    LeftUpperNode = np.where(np.logical_and(np.abs(Node[:, 0] - 0) < eps, np.abs(Node[:, 1] - 1) < eps))[0]
    RightBottomNode = np.where(np.logical_and(np.abs(Node[:, 0] - 3) < eps, np.abs(Node[:, 1] - 0) < eps))[0]
    FixedNodes = np.concatenate((LeftEdgeNodes, RightBottomNode))
    Supp = np.zeros((len(FixedNodes), 3), dtype=int)
    Supp[:, 0] = FixedNodes
    Supp[:-1, 1] = 1
    Supp[-1, 2] = 1
    Load = np.zeros((1, 3))
    Load[0, 0], Load[0, 1], Load[0, 2] = LeftUpperNode[0], 0, -0.5
    x = [Supp, Load]
    return x


MbbDomain = TopologicalDomain("Mbb Domain", [-0.5, 3.5, -0.5, 1.5], _mbbsdf, _mbbbc)
