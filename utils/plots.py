import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook')
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True

def plot_golden_section(x, y, z, golden_sec):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    fig.delaxes(ax[0])
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].view_init(40, 20)
    ax[0].plot_surface(x, y, z, cmap='viridis', linewidth =0)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_zlabel(r"$f(x_{1}, x_{2})$")
    ax[0].set_title(r"Superfície da função $f(x_{1}, x_{2})$")

    ax[1].contour(x, y, z, 200, cmap='viridis')
    ax[1].plot(golden_sec[0][0], golden_sec[0][1], color='red', markersize=10, marker='x', label = f'Ponto Mínimo')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title(r"Curvas de nível da função $f(x_{1}, x_{2})$")
    ax[1].legend(loc = 'upper right')

    #plt.tight_layout()
    plt.show()