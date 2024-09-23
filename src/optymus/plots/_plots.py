import warnings

warnings.filterwarnings('ignore')

import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style('whitegrid')


def plot_function(f_obj, min=-10, max=10, n=100, n_levels=50, show=True):
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
    Z = f_obj([X, Y])

    # 3D Surface plot
    surface = go.Surface(z=Z, x=X, y=Y, colorscale='PuBuGn_r', showscale=False)

    # Contour plot for objective function
    contour_obj = go.Contour(
        z=Z, x=x, y=y, 
        colorscale='PuBuGn_r',
        ncontours=n_levels,
        contours={'showlabels': False},
        showscale=False,
        line={'color': 'white'},
        line_smoothing=0.85,
        name='Objective Function'
    )

    fig = sp.make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'xy'}]],
                           subplot_titles=("Function Surface f(x1, x2)", "Contour Plot f(x1, x2)"))

    fig.add_trace(surface, row=1, col=1)
    fig.add_trace(contour_obj, row=1, col=2)

    fig.update_layout(
        scene={
            'xaxis_title': 'x1', 'yaxis_title': 'x2', 'zaxis_title': 'f(x1, x2)',
            'camera': {'eye': {'x': 1.87, 'y': 0.88, 'z': 1.00}}
        },
        xaxis_title='x1',
        yaxis_title='x2',
        width=800,
        height=300,
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
        template='seaborn'
    )

    if show:
        fig.show()

def plot_optim(f_obj=None, f_cons=None, x0=None, method=None, path=True, comparison=None, print_opt=False, renderer='notebook', show=True, template='seaborn', min=-10, max=10, n=100, n_levels=50):
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
        Z = f_obj([X, Y])
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
                Z[i, j] = f_obj(pca.inverse_transform([X[i, j], Y[i, j]]))

    # 3D Surface plot
    surface = go.Surface(z=Z, x=X, y=Y, colorscale='PuBuGn_r', showscale=False)

    # Contour plot for objective function
    contour_obj = go.Contour(
        z=Z, x=x, y=y, colorscale='PuBuGn_r',
        contours={'showlabels': False},
        ncontours=n_levels,
        line={'color': 'white'},
        showscale=False,
        line_smoothing=0.85,
        name='Objective Function'
    )

    # Initial and optimal points
    initial_point = go.Scatter(
        x=[x0[0]], y=[x0[1]], mode='markers', marker={'color': 'black', 'size': 10, 'symbol': 'diamond-open'},
        name=f'Initial Point - {x0}'
    )

    if method is not None:
        optimal_point = go.Scatter(
            x=[method['xopt'][0]], y=[method['xopt'][1]], mode='markers', marker={'color': 'tomato', 'size': 10, 'symbol': 'x'},
            name=f'Optimal Point ({method["num_iter"]} iter.)'
        )

        # Optimization path
        optimization_path = go.Scatter(
            x=method['path'][:, 0], y=method['path'][:, 1], mode='lines+markers', line={'color': 'black'},
            name='Optimization Path',
        )

    # Create subplot
    fig = sp.make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'xy'}]],
                           subplot_titles=("Function Surface f(x1, x2)", "Contour Plot f(x1, x2)"))

    fig.add_trace(surface, row=1, col=1)
    fig.add_trace(contour_obj, row=1, col=2)
    fig.add_trace(initial_point, row=1, col=2)
    if path is True and comparison is not None:
        for method in comparison['path_dict'].keys():
            path_x = [p[0] for p in comparison['path_dict'][method]]
            path_y = [p[1] for p in comparison['path_dict'][method]]
            fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines+markers', name=method), row=1, col=2)

    if path is True and comparison is None:
        fig.add_trace(optimization_path, row=1, col=2)
        fig.add_trace(optimal_point, row=1, col=2)

    if f_cons is not None:
        colors_map = px.colors.named_colorscales()
        for c, colors, i in zip(f_cons, colors_map, range(len(f_cons))):
            contour_constr = go.Contour(
                x=x, y=y,
                z=c([X, Y]),
                contours_coloring='lines',
                colorscale=colors,
                showscale=True,
                contours={
                    'start': 0, 'end': 0, 'size': 1,
                    'showlabels': False,
                },
                ncontours=n_levels,
                line={'dash': 'dash', 'color': 'red', 'width': 2, 'smoothing': 0.85},
                name=f'Constraint {i+1}', showlegend=True
            )
            fig.add_trace(contour_constr, row=1, col=2)

    if print_opt:
        fig.add_annotation(
            x=method['xopt'][0], y=method['xopt'][1],
            text=f'{method["xopt"][0]:.4f}, {method["xopt"][1]:.4f}',
            showarrow=True, arrowhead=2, ax=40, ay=-30, arrowcolor='white',
            bordercolor='white', borderwidth=1, borderpad=2, bgcolor='white',
            opacity=0.8, font={'color': 'black', 'size': 12},
        )

    fig = go.Figure(fig)
    fig.update_layout(
        scene={
            'xaxis_title': 'x1', 'yaxis_title': 'x2', 'zaxis_title': 'f(x1, x2)',
            'camera': {'eye': {'x': 1.87, 'y': 0.88, 'z': 1.00}}
        },
        xaxis_title='x1',
        yaxis_title='x2',
        width=800,
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
        template=template
    )

    if show is True:
        fig.show(renderer=renderer)