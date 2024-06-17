import jax
import pandas as pd
from optymus.methods import (
    adagrad,
    adam,
    adamax,
    bfgs,
    conjugate_gradient,
    gradient_descent,
    newton_raphson,
    powell,
    rmsprop,
    univariant,
    yogi,
)
from optymus.plots import plot_alphas, plot_optim

jax.config.update("jax_enable_x64", True)

METHODS = {
    "univariant": univariant,
    "powell": powell,
    "gradient_descent": gradient_descent,
    "conjugate_gradient": conjugate_gradient,
    "bfgs": bfgs,
    "newton_raphson": newton_raphson,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "adam": adam,
    "adamax": adamax,
    "yogi": yogi,
}


class Optimizer:
    def __init__(self, f_obj=None, f_cons=None, x0=None, method='gradient_descent', **kwargs):
        """
        Initializes the Optimizer class.

        Args:
            f_obj (function): The objective function to be minimized.
            x0 (np.ndarray): The initial guess for the minimum.
            method (str, optional): The optimization method to use. Defaults to 'gradient_descent'.
        """
        self.f_obj = f_obj
        self.f_cons = f_cons
        self.x0 = x0
        self.method = method

        if self.method not in METHODS:
            msg = f"Method '{method}' not available. Available methods: {list(METHODS.keys())}"
            raise ValueError(msg)

        if self.f_obj is None:
            msg = "Objective function is required."
            raise ValueError(msg)

        if self.x0 is None:
            msg = "Initial guess is required."
            raise ValueError(msg)

        # Run the optimization and store results
        self.opt = METHODS[self.method](f_obj=self.f_obj,f_cons=self.f_cons, x0=self.x0,**kwargs)

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
            "Initial Guess": [self.opt.get('x0', 'N/A')],
            "Optimal Solution": [self.opt.get('xopt', 'N/A')],
            "Objective Function Value": [self.opt.get('fmin', 'N/A')],
            "Number of Iterations": [self.opt.get('num_iter', 'N/A')],
            "Time Elapsed": [self.opt.get('time', 'N/A')],
        }

        return pd.DataFrame(table_data, index=['Optimization Results'])

    def plot_results(self, **kwargs):
        """Plots the optimization path and function surface."""
        plot_optim(f_obj=self.f_obj, f_cons=self.f_cons, x0=self.x0, method=self.opt, **kwargs)

    def create_dashboard(self, port=8050, **kwargs):
        """Generates a Dash dashboard with optimization results."""

        # if dash is not installed, install it
        try:
            import dash
            import dash_bootstrap_components as dbc
            from dash import Input, Output, dcc, html
            from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template

        except ImportError:
            import os
            os.system("pip install -q dash")
            os.system("pip install -q dash-bootstrap-components")
            os.system("pip install -q dash-bootstrap-templates")
            import dash
            import dash_bootstrap_components as dbc
            from dash import Input, Output, dcc, html
            from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template

        dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
        app = dash.Dash(__name__, title="optymus", external_stylesheets=[dbc.themes.FLATLY, dbc_css])

        theme_switch = ThemeSwitchAIO(
            aio_id="theme", themes=[dbc.themes.FLATLY, dbc.themes.SLATE]
        )
        if self.f_cons is not None:
            self.method = f"{self.method} with constraints"

        self.path = self.opt.get('path', None)

        fig_optim = dcc.Graph(id='contour-plot')
        fig_alpha = dcc.Graph(id='alpha-plot')

        navbar = html.H4(
            "Optymus Dashboard", className="bg-primary text-white p-2 mb-2 text-left",
        )

        footer = dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.P(
                        [
                            html.Span("optymus", className="mr-2"),
                            html.A(
                                html.Span("GitHub", className="text-white"),
                                href="https://github.com/quant-sci/optymus",
                                target="_blank",
                            ),
                        ],
                        className="lead",
                    )
                )
            )
        )

        # Create Dash layout
        app.layout = html.Div(children=[
            navbar,
            theme_switch,
            html.Br(),

            dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Function surface and countour", className="card-title"),
                                fig_optim,
                            ]
                        )
                    )
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Optimization Details", className="card-title"),
                                dbc.Table(
                                    [
                                        html.Tr([html.Th("Parameter"), html.Th("Value")]),
                                        html.Tr([html.Td("Method"), html.Td(str(self.method))]),
                                        html.Tr([html.Td("Final Solution"), html.Td(str(self.opt.get('xopt', 'N/A')))]),
                                        html.Tr([html.Td("Objective Function Value"), html.Td(str(self.opt.get('fmin', 'N/A')))]),
                                        html.Tr([html.Td("Number of Iterations"), html.Td(str(self.opt.get('num_iter', 'N/A')))]),
                                        html.Tr([html.Td("Initial Guess"), html.Td(str(self.x0))]),
                                        html.Tr([html.Td("Time Elapsed"), html.Td(str(self.opt.get('time', 'N/A')))]),
                                    ],
                                    bordered=True,
                                )
                            ]
                        )
                    )
                ),
            ],
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            fig_alpha,
                                        ]
                                    )
                                ]
                            ),
                        )
                    ),
                ],
            ),
            html.Br(),
            footer,
        ])

        @app.callback(
            Output("contour-plot", "figure"),
            Output("alpha-plot", "figure"),
            Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
        )

        def update_theme(toggle):
            theme = "flatly" if toggle else "slate"
            load_figure_template(theme)
            return plot_optim(f_obj=self.f_obj, f_cons=self.f_cons, x0=self.x0, method=self.opt, path=True, template=theme, **kwargs), \
                   plot_alphas(self.opt.get('alphas', None), template=theme)


        # Run the Dash app
        app.run_server(port=port, debug=False, use_reloader=False)

        # open the browser
        import webbrowser
        webbrowser.open(f'http://localhost:{port}')

