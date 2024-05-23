# Full path: optymus/optymus/optimizer/_optimize.py

import dash
import dash_bootstrap_components as dbc
import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from optymus.methods import (
    adagrad,
    adam,
    adamax,
    bfgs,
    conjugate_gradients,
    gradient_descent,
    l_bfgs,
    newton_raphson,
    powell,
    rmsprop,
    univariant,
)

jax.config.update("jax_enable_x64", True)

METHODS = {
    "univariant": univariant,
    "powell": powell,
    "gradient_descent": gradient_descent,
    "conjugate_gradients": conjugate_gradients,
    "bfgs": bfgs,
    "l_bfgs": l_bfgs,
    "newton_raphson": newton_raphson,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "adam": adam,
    "adamax": adamax
}


class Optimizer:
    def __init__(self, f_obj, x0, method='gradient_descent', tol=1e-5, max_iter=100):
        """
        Initializes the Optimizer class.

        Args:
            f_obj (function): The objective function to be minimized.
            x0 (np.ndarray): The initial guess for the minimum.
            method (str, optional): The optimization method to use. Defaults to 'gradient_descent'.
            tol (float, optional): The tolerance for convergence. Defaults to 1e-5.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        """
        self.f_obj = f_obj
        self.x0 = x0
        self.method = method
        self.tol = tol
        self.max_iter = max_iter

        if self.method not in METHODS:
            msg = f"Method '{method}' not available. Available methods: {list(METHODS.keys())}"
            raise ValueError(msg)

        # Run the optimization and store results
        self.opt = METHODS[self.method](f_obj=self.f_obj, x0=self.x0, tol=self.tol)

    def check_dimension(self):
        """Returns the dimension of the problem."""
        return len(self.x0)

    def get_results(self):
        """Returns the optimization results dictionary."""
        return self.opt

    def print_report(self):
        """Prints a formatted summary of the optimization results."""
        table_data = {
            "Method": [self.method],
            "Initial Guess": [self.x0],
            "Optimal Solution": [self.opt.get('xopt', 'N/A')],
            "Objective Function Value": [self.opt.get('fmin', 'N/A')],
            "Number of Iterations": [self.opt.get('num_iter', 'N/A')],
        }

        return pd.DataFrame(table_data, index=['Optimization Results'])

    def create_dashboard(self, port=8050):
        """Generates a Dash dashboard with optimization results."""

        app = dash.Dash(__name__, title="optymus", external_stylesheets=[dbc.themes.FLATLY])

        # Create 3D surface data (if applicable)
        dimension = 2

        if self.check_dimension() == dimension:
            x_values = np.linspace(-3, 3, 100)
            y_values = np.linspace(-3, 3, 100)
            x_grid, y_grid = np.meshgrid(x_values, y_values)
            z_grid = self.f_obj([x_grid, y_grid])

        self.path = self.opt.get('path', None)

        navbar = html.H4(
            "Optymus Dashboard", className="bg-primary text-white p-2 mb-2 text-left"
        )

        # Create Dash layout
        app.layout = html.Div(children=[
            navbar,

            # Optimization path plot (only if 2D - adapted for 3D surface)
            dbc.Row(
            [
                dbc.Col(dbc.Card(dbc.CardBody([html.Div(dcc.Graph(
                                            id='path-graph',
                                            figure={
                                                'data': [
                                                    go.Surface(
                                                        x=x_grid,
                                                        y=y_grid,
                                                        z=z_grid,
                                                        colorscale='Viridis',
                                                        opacity=0.8
                                                    ),
                                                    go.Scatter3d(
                                                        x=[],
                                                        y=[],
                                                        z=[],
                                                        mode='lines+markers',
                                                        marker={"size": 5, "color": 'red'},
                                                        line={"width": 3, "color": 'red'}
                                                    )
                                                ],
                                                'layout': go.Layout(
                                                    title='Objective Function Surface',
                                                    scene={
                                                        "xaxis_title": 'X',
                                                        "yaxis_title": 'Y',
                                                        "zaxis_title": 'Objective Function Value'
                                                    },
                                                    margin={"l": 0, "r": 0, "b": 0, "t": 40}
                                                )
                                            },

                                        ),
                                    ),
                                ],) if self.check_dimension() == 2 else html.Div(),

                                        )),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div(dcc.Graph(
                                        id='convergence-graph',
                                        figure={
                                            'data': [
                                                go.Contour(
                                                    z=z_grid,
                                                    x=x_values,
                                                    y=y_values,
                                                    colorscale='Viridis',
                                                    opacity=0.8,
                                                    contours={"showlabels": True},
                                                ),
                                                go.Scatter(
                                                    x=[i[0] for i in self.path],
                                                    y=[i[1] for i in self.path],
                                                    mode='lines+markers',
                                                    marker={"size": 5, "color": 'red'},
                                                    line={"width": 3, "color": 'red'}
                                                )
                                            ],
                                            'layout': go.Layout(
                                                title='Path of Optimization',
                                                margin={"l": 0, "r": 0, "b": 0, "t": 40}
                                           )
                                            },
                                        ),
                                    ),
                                ],
                            ),
                        ),
                    ),
                dbc.Col(
                    html.Div(
                        dbc.Table(
                            children=[
                                html.Tr([html.Th("Parameter"), html.Th("Value")]),
                                html.Tr([html.Td("Method"),
                                            html.Td(str(self.method))]),
                                html.Tr([html.Td("Final Solution"),
                                            html.Td(str(self.opt.get('xopt', 'N/A')))]),
                                html.Tr([html.Td("Objective Function Value"),
                                            html.Td(str(self.opt.get('fmin', 'N/A')))]),
                                html.Tr([html.Td("Number of Iterations"),
                                            html.Td(str(self.opt.get('num_iter', 'N/A')))]),
                                html.Tr([html.Td("Initial Guess"),
                                            html.Td(str(self.x0))]),
                            ],
                            bordered=True,
                        )
                    )
                ),
            ],
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Markdown(
                                id='pseudocode',
                                children=f"""
                                ```python
                                def {self.method.lower()}(f_obj, x0):
                                    # Pseudocode for {self.method}
                                    # ...
                                ```
                                """
                            )
                        )
                    ),
                ],
            ),
        ])

        # Run the Dash app
        app.run_server(port=port, debug=False, use_reloader=False)

        # open the browser
        import webbrowser
        webbrowser.open(f'http://localhost:{port}')
