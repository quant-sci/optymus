import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from qsopt.optim import method_optim


sns.set_style('whitegrid')
sns.set_context('notebook')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True


def plot_optim(f, x0, method, path=True):
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
    ax[0].set_title(r"Superfície da função $f(x_{1}, x_{2})$")

    contour = ax[1].contour(X, Y, Z, 200, cmap='cividis')
    ax[1].plot(x0[0], x0[1], color='green', marker='o', label=f'Ponto Inicial - {x0}')
    ax[1].scatter(method['xopt'][0], method['xopt'][1], color='red', s=50, marker='o', label=f'Ponto Mínimo - num_iter: {method["num_iter"]}')
    ax[1].text(method['xopt'][0] + 1, method['xopt'][1] - 2, f'{method["xopt"][0]}, \n{method["xopt"][1]}', size=10, zorder=2, color='k', bbox=dict(boxstyle="round", alpha=1, facecolor='white'))    
    if path == True:
        ax[1].plot(method['path'][:, 0], method['path'][:, 1], '-k', label='Caminho da otimização')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_title(method['method_name'])
    ax[1].legend(loc='upper right')
    plt.show()


def plot_countour(f, x0, method, ax, path=True, box_size=100):
    x = np.linspace(-box_size, box_size, 100)
    y = np.linspace(-box_size, box_size, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])

    contour = ax.contour(X, Y, Z, 200, cmap='cividis')
    ax.plot(x0[0], x0[1], color='green', marker='o', label=f'Ponto Inicial - {x0}')
    ax.scatter(method['xopt'][0], method['xopt'][1], color='red', s=50, marker='o', label=f'Ponto Mínimo - num_iter: {method["num_iter"]}')
    ax.text(method['xopt'][0] + 1, method['xopt'][1] - 2, f'{method["xopt"][0]}, \n{method["xopt"][1]}', size=10, zorder=2, color='k', bbox=dict(boxstyle="round", alpha=1, facecolor='white'))    
    if path == True:
        ax.plot(method['path'][:, 0], method['path'][:, 1], '-k', label='Caminho da otimização')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(method['method_name'])
    ax.legend(loc='upper right')

def plot_methods(f, x0, grad, hess):
    METHODS_OPTIM = {
    'univariant': method_optim(method_name='univariant', objective_function=f, gradient=grad, initial_point=x0),
    'powell': method_optim(method_name='powell', objective_function=f, initial_point=x0, gradient=grad),
    'steepest_descent': method_optim(method_name='steepest_descent', objective_function=f, gradient=grad, initial_point=x0),
    'fletcher_reeves': method_optim(method_name='fletcher_reeves', objective_function=f, gradient=grad, initial_point=x0),
    'bfgs': method_optim(method_name='bfgs', objective_function=f, gradient=grad, initial_point=x0),
    'newton_raphson': method_optim(method_name='newton_raphson', objective_function=f, gradient=grad, hessian=hess, initial_point=x0),
    }
    methods = ['univariant', 'powell', 'steepest_descent', 'fletcher_reeves', 'bfgs', 'newton_raphson']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, m in enumerate(methods):
        method = METHODS_OPTIM[m]
        plot_countour(f, x0, method, ax=axes[i], path = True)
    plt.tight_layout()