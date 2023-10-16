import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True


def plot_optimization(x, y, z, method, path=None):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    fig.delaxes(ax[0])
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].view_init(40, 20)
    ax[0].plot_surface(x, y, z, cmap='viridis', linewidth =0)
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_zlabel(r"$f(x_{1}, x_{2})$")
    ax[0].set_title(r"Superfície da função $f(x_{1}, x_{2})$")

    ax[1].contour(x, y, z, 200, cmap='viridis')
    ax[1].plot(method['initial_point'][0], method['initial_point'][1], color='green', markersize=7, marker='o', label = f'Ponto Inicial')
    ax[1].text(method['initial_point'][0]+1, method['initial_point'][1], f'{method["initial_point"][0]}, {method["initial_point"][1]}', size=10, zorder=2, color='k', bbox=dict(boxstyle="round", alpha=1, facecolor='white'))
    ax[1].plot(method['xopt'][0], method['xopt'][1], color='red', markersize=8, marker='o', label = f'Ponto Mínimo - num_iter: {method["num_iter"]}')
    ax[1].text(method['xopt'][0]+1, method['xopt'][1]-2, f'{method["xopt"][0]}, \n{method["xopt"][1]}', size=10, zorder=2, color='k', bbox=dict(boxstyle="round", alpha=1, facecolor='white'))
    ax[1].plot([method['initial_point'][0], method['xopt'][0]], [method['initial_point'][1], method['xopt'][1]], color='black', linestyle='solid')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_title(r"Curvas de nível da função $f(x_{1}, x_{2})$")
    ax[1].legend(loc = 'upper right')

    #plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()