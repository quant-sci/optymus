import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('notebook')


def plot_optim(f, x0, method, path=True):
    """
    Plot the optimization path and the function surface.
    
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
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))

    fig.delaxes(ax[0])
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].view_init(40, 20)
    ax[0].plot_surface(X, Y, Z, cmap='cividis', linewidth =0)
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_zlabel(r"$f(x_{1}, x_{2})$")
    ax[0].set_title(r"Function surface $f(x_{1}, x_{2})$")

    contour = ax[1].contour(X, Y, Z, 200, cmap='cividis')
    ax[1].plot(x0[0], x0[1], color='green', marker='o', label=f'Initial Point - {x0}')
    ax[1].scatter(method['xopt'][0], method['xopt'][1], color='red', s=50, marker='o', label=f'Optimal point ({method["num_iter"]} iter.)')
    ax[1].text(method['xopt'][0] + 1, method['xopt'][1] - 2, f'{method["xopt"][0]}, \n{method["xopt"][1]}', size=10, zorder=2, color='k', bbox=dict(boxstyle="round", alpha=1, facecolor='white'))    
    if path == True:
        ax[1].plot(method['path'][:, 0], method['path'][:, 1], '-k', label='Optimization path')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_title(method['method_name'])
    ax[1].legend(loc='upper right')
    plt.show()