from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg") # pour ne pas appler Qt qui fonctionne mal avec Wayland
mpl.rcParams['figure.dpi'] = 175      # augmente la résolution d'affichage
mpl.rcParams['figure.figsize'] = (8, 6)  # taille par défaut des figures (en pouces)





dune_path = Path("Toma/double_sin.txt") #ATTENTION ICI DOUBLE SIN
if not dune_path.exists():
    dune_path = Path("./../double_sin.txt")


data = np.loadtxt(dune_path)

x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)


def gradient_tpp(data, dx=0.5, dy=0.5):
    """O'Neill & Mark (TPP)
     différences finies avant : voisin du dessus (y) et à droite (x)."""
    dzdx = np.empty_like(data)
    dzdy = np.empty_like(data)

    dzdx[:, :-1] = (data[:, 1:] - data[:, :-1]) / dx
    dzdx[:, -1] = dzdx[:, -2]

    dzdy[:-1, :] = (data[1:, :] - data[:-1, :]) / dy
    dzdy[-1, :] = dzdy[-2, :]

    return np.sqrt(dzdx**2 + dzdy**2)


def gradient_fcn(data, dx=0.5, dy=0.5):
    """FCN, différences centrales sur les 4 plus proches voisins."""
    dzdx = np.empty_like(data)
    dzdy = np.empty_like(data)

    dzdx[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2 * dx)
    dzdx[:, 0]    = (data[:, 1]  - data[:, 0])   / dx
    dzdx[:, -1]   = (data[:, -1] - data[:, -2])  / dx

    dzdy[1:-1, :] = (data[2:, :] - data[:-2, :]) / (2 * dy)
    dzdy[0, :]    = (data[1, :]  - data[0, :])   / dy
    dzdy[-1, :]   = (data[-1, :] - data[-2, :])  / dy

    return np.sqrt(dzdx**2 + dzdy**2)


def gradient_evans(data, dx=0.5, dy=0.5):
    """Evans, estimation du gradient sur un voisinage 3*3."""
    p = np.zeros_like(data)
    q = np.zeros_like(data)

    # Partie intérieure
    p[1:-1, 1:-1] = (
        data[:-2, 2:] + data[1:-1, 2:] + data[2:, 2:]
        - data[:-2, :-2] - data[1:-1, :-2] - data[2:, :-2]
    ) / (6 * dx)

    q[1:-1, 1:-1] = (
        data[2:, :-2] + data[2:, 1:-1] + data[2:, 2:]
        - data[:-2, :-2] - data[:-2, 1:-1] - data[:-2, 2:]
    ) / (6 * dy)

    # Bords : différences finies avant/arrière pour p
    p[:, 0]    = (data[:, 1]  - data[:, 0])   / dx
    p[:, -1]   = (data[:, -1] - data[:, -2])  / dx
    p[0, 1:-1] = p[1, 1:-1]
    p[-1, 1:-1] = p[-2, 1:-1]

    # Bords : différences finies avant/arrière pour q
    q[0, :]    = (data[1, :]  - data[0, :])   / dy
    q[-1, :]   = (data[-1, :] - data[-2, :])  / dy
    q[1:-1, 0]  = q[1:-1, 1]
    q[1:-1, -1] = q[1:-1, -2]

    return np.sqrt(p**2 + q**2)


# Plot de comparaison
grad_tpp   = data-gradient_tpp(data)
#ATTENTION ICI ON FAIT LA DIFFERENCE AVEC L'OG
grad_fcn   = data-gradient_fcn(data)
grad_evans = data-gradient_evans(data)

grad_vmin = min(grad_tpp.min(), grad_fcn.min(), grad_evans.min())
grad_vmax = max(grad_tpp.max(), grad_fcn.max(), grad_evans.max()) / 4
grad_levels = np.linspace(grad_vmin, grad_vmax, 101)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Dunkerque — comparaison des méthodes de gradient', fontsize=13)

titles   = ['MNT original', "Gradient TPP\n(O'Neill & Mark)", 'Gradient FCN\n(4 voisins)', 'Gradient Evans']
datasets = [data, grad_tpp, grad_fcn, grad_evans]

extent = [x[0], x[-1], y[0], y[-1]]

ax_mnt = axes.flat[0]
im_mnt = ax_mnt.imshow(data, origin='lower', extent=extent, aspect='auto', cmap='gist_earth')
ax_mnt.set_title(titles[0])
ax_mnt.set_xlabel('x')
ax_mnt.set_ylabel('y')
fig.colorbar(im_mnt, ax=ax_mnt, label='Altitude [m]')

for ax, title, d in zip(list(axes.flat)[1:], titles[1:], [grad_tpp, grad_fcn, grad_evans]):
    im = ax.imshow(d, origin='lower', extent=extent, aspect='auto',
                   cmap='seismic', vmin=grad_vmin, vmax=grad_vmax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, label='|∇z| [m/m]')

plt.tight_layout()
plt.show()
