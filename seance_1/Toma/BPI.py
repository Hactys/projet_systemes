from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import convolve


def _disk_kernel(r):
    D = 2 * r + 1
    kernel = np.zeros((D, D))
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if 0 < di**2 + dj**2 <= r**2:
                kernel[r + di, r + dj] = 1.0
    return kernel


def moyenne_voisins_disque(mat, r=3):
    kernel = _disk_kernel(r)
    k = int(kernel.sum())

    mat_filled = np.where(np.isnan(mat), 0.0, mat)
    valid = (~np.isnan(mat)).astype(float)

    sum_vals = convolve(mat_filled, kernel, mode="constant", cval=0.0)
    count_vals = convolve(valid, kernel, mode="constant", cval=0.0)

    # Moyenne uniquement si le disque est entièrement valide (même condition qu'avant)
    moy = np.where(np.round(count_vals) == k, sum_vals / k, np.nan)
    return moy


def calcul_BPI(mnt, r=2):
    """BPI = valeur pixel - moyenne des voisins dans le disque."""
    return mnt - moyenne_voisins_disque(mnt, r)


def load(name):
    p = Path(name)
    if not p.exists():
        p = Path("..") / name
    return np.loadtxt(p)


# ── affichage ─────────────────────────────────────────────────────────────────
if __name__=="__main__":
    datasets = [load("sin_card.txt"), load("plateau.txt"),
            load("plan.txt"),     load("double_sin.txt")]
    noms     = ["sin_card", "plateau", "plan", "double_sin"]
    RAYON    = 2
    cmap     = plt.cm.RdBu_r

    # Calcul du vmax global sur tous les jeux de données
    all_bpis = [calcul_BPI(data, RAYON) for data in datasets]
    vmax_global = max(
        float(np.nanpercentile(np.abs(bpi[~np.isnan(bpi)]), 98))
        for bpi in all_bpis
    ) or 1e-9
    norm = TwoSlopeNorm(vmin=-vmax_global, vcenter=0, vmax=vmax_global)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"BPI sur valeurs MNT (rayon={RAYON})", fontsize=14, fontweight="bold")

    for ax, nom, bpi in zip(axes, noms, all_bpis):

        im = ax.imshow(bpi, origin="lower", cmap=cmap, norm=norm)  # norm commune
        ax.set_title(nom, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label="BPI")

    plt.tight_layout()
    plt.show()
