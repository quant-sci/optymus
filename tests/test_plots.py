import numpy as np
from unittest.mock import patch
import jax.numpy as jnp
from plotly.graph_objs import Figure

from optymus.plots import plot_function, plot_alphas, plot_optim

# Define a simple quadratic function for testing
def quadratic(x):
    return x[0] ** 2 + x[1] ** 2

def test_plot_function():
    with patch('matplotlib.pyplot.show'):
        plot_function(quadratic, title="Quadratic Function")

def test_plot_alphas():
    alphas = np.linspace(0.1, 1.0, 10)
    fig = plot_alphas(alphas)
    assert isinstance(fig, Figure), "Output should be a Plotly Figure object"

def test_plot_optim():
    x0 = jnp.array([0.0, 1.0])
    method = {
        'xopt': np.array([0.0, 0.0]),
        'num_iter': 100,
        'path': np.array([[0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    }
    
    fig = plot_optim(f_obj=quadratic, x0=x0, method=method, path=True, show=False)
    assert isinstance(fig, Figure), "Output should be a Plotly Figure object"
