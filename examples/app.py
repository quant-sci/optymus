import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go

# Define the Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Gradient of the Rosenbrock function
def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Gradient descent algorithm
def gradient_descent(start, learning_rate, num_steps):
    path = [start]
    x, y = start
    for _ in range(num_steps):
        grad = rosenbrock_grad(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        path.append((x, y))
    return np.array(path)

# Initial parameters
start = (-1.5, 2)
learning_rate = 0.001
num_steps = 100

# Compute the optimization path
path = gradient_descent(start, learning_rate, num_steps)

# Generate data for the contour plot
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Create the contour plot
contour = go.Contour(z=Z, x=x, y=y, colorscale='Viridis', opacity=0.7)

# Create the initial plot with the starting point
initial_point = go.Scatter(x=[start[0]], y=[start[1]], mode='markers', marker=dict(color='red', size=10), name='Start')

# Create the initial figure
fig = go.Figure(data=[contour, initial_point])

# Create frames for the animation
frames = []
for k in range(1, len(path)):
    frames.append(go.Frame(
        data=[
            go.Scatter(x=path[:k, 0], y=path[:k, 1], mode='lines+markers', marker=dict(color='red', size=5)),
            contour  # Ensure the contour plot is included in each frame
        ]
    ))

# Update the figure with frames
fig.frames = frames

# Add animation controls
fig.update_layout(
    updatemenus=[dict(type='buttons', showactive=False,
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, {'frame': {'duration': 50, 'redraw': True},
                                                 'fromcurrent': True}])])],
    xaxis=dict(range=[-2, 2], autorange=False),
    yaxis=dict(range=[-1, 3], autorange=False)
)

# Create Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Optimization Path Animation"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8040)
