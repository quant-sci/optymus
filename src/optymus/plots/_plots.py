import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style('whitegrid')


def plot_function(f, title=None, min=-10, max=10, n=100):
    """
    Plot the function surface.

    Parameters
    ----------
    f : callable
        Function to be plotted.

    Returns
    -------
    Plot of the function surface.
    """
    x = np.linspace(min, max, n)
    y = np.linspace(min, max, n)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))

    fig.delaxes(ax[0])
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].view_init(40, 20)
    ax[0].plot_surface(X, Y, Z, cmap='viridis', linewidth =0)
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_zlabel("f(x_1, x2)")
    ax[0].set_title(f'{title} surface f(x1, x2)')

    countour = ax[1].contour(X, Y, Z, 200, cmap='viridis')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    if title is not None:
        ax[1].set_title(f'{title} contour f(x1, x2)')
    else:
        ax[1].set_title("Function contour f(x1, x2)")
    plt.tight_layout()
    plt.show()

def plot_optim(f, x0, method, path=True, print_opt=True, show=True, min=-10, max=10, n=100):
    """
    Plot the optimization path and the function surface using Plotly.

    Parameters
    ----------
    f : callable
        Function to be optimized.
    x0 : array_like
        Initial guess.
    method : dict
        Dictionary containing the optimization method information.
    path : bool, optional
        If True, plot the optimization path.
        Default is True.

    Returns
    -------
    Plot of the function surface and the optimization path.
    """
    if len(x0) == 2:
        x = np.linspace(min, max, n)
        y = np.linspace(min, max, n)
        X, Y = np.meshgrid(x, y)
        Z = f([X, Y])
    else:
        np.random.seed(42)
        pca = PCA(n_components=2)
        pca.fit(np.random.rand(100, len(x0)))
        x = np.linspace(min, max, n)
        y = np.linspace(min, max, n)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(pca.inverse_transform([X[i, j], Y[i, j]]))

    # 3D Surface plot
    surface = go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', showscale=False)

    # Contour plot
    contour = go.Contour(z=Z, x=x, y=y, colorscale='Viridis',
                         contours={'showlabels': True}, showscale=False,
                         line_smoothing=0.85)

    # Initial and optimal points
    initial_point = go.Scatter(
        x=[x0[0]], y=[x0[1]], mode='markers', marker={'color': 'green', 'size': 10},
        name=f'Initial Point - {x0}'
    )

    optimal_point = go.Scatter(
        x=[method['xopt'][0]], y=[method['xopt'][1]], mode='markers', marker={'color': 'red', 'size': 10},
        name=f'Optimal Point ({method["num_iter"]} iter.)'
    )

    # Optimization path
    if path:
        optimization_path = go.Scatter(
            x=method['path'][:, 0], y=method['path'][:, 1], mode='lines+markers', line={'color': 'red'},
            name='Optimization Path'
        )
    else:
        optimization_path = None

    # Create subplot
    fig = sp.make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'xy'}]],
                           subplot_titles=("Function Surface f(x1, x2)", method['method_name']))

    fig.add_trace(surface, row=1, col=1)
    fig.add_trace(contour, row=1, col=2)
    fig.add_trace(initial_point, row=1, col=2)
    fig.add_trace(optimal_point, row=1, col=2)

    if print_opt:
        fig.add_annotation(
            x=method['xopt'][0], y=method['xopt'][1],
            text=f'{method["xopt"][0]:.4f}, {method["xopt"][1]:.4f}',
            showarrow=True, arrowhead=2, ax=40, ay=-30, arrowcolor='white',
            bordercolor='white', borderwidth=1, borderpad=2, bgcolor='white'
        )

    if optimization_path:
        fig.add_trace(optimization_path, row=1, col=2)

    fig = go.Figure(fig)
    fig.update_layout(
        scene={
            'xaxis_title': 'x1', 'yaxis_title': 'x2', 'zaxis_title': 'f(x1, x2)',
            'camera': {'eye': {'x': 1.87, 'y': 0.88, 'z': 1.00}}
        },
        xaxis_title='x1',
        yaxis_title='x2',
        width=900,
        height=400,
        legend={
            'orientation': 'h',
            'x': 0.5, 'xanchor': 'center',
            'y': -0.25, 'yanchor': 'top'
        },
        margin={
            'l': 5,
            'r': 5,
            't': 50,
            'b': 0},
    )

    if show:
        fig.show()

    return fig
