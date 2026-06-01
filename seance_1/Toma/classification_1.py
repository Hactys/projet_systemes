from enum import Enum
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from BPI import calcul_BPI
from pentes_Toma import pente, Evans


class Terrain:
    PLAT = 0
    DEPRESSION = 1
    CRETE = 3
    PENTE = 2
    RIDULE = 4
    LISSE = 5


def classif_1(mnt):
    bpi = calcul_BPI(mnt, r=50)
    classe = np.zeros_like(mnt)
    pts = pente(*Evans(mnt))
    # parcourir les bpi pour trier les points
    for i in range(bpi.shape[0]):
        for j in range(bpi.shape[1]):
            if np.isnan(bpi[i, j]):
                classe[i, j] = -1  # bord ou point invalide
            elif bpi[i, j] <= -0.4:
                classe[i, j] = Terrain.DEPRESSION
            elif bpi[i, j] >= 0.5:
                classe[i, j] = Terrain.PENTE
            else:
                if pts[i, j] < 0.18:
                    classe[i, j] = Terrain.PLAT
                else:
                    classe[i, j] = Terrain.CRETE
    return classe


def classif_2(mnt):
    bpi = calcul_BPI(mnt, r=5)
    classe = np.zeros_like(mnt)
    # parcourir les bpi pour trier les points
    for i in range(bpi.shape[0]):
        for j in range(bpi.shape[1]):
            if np.isnan(bpi[i, j]):
                classe[i, j] = -1  # bord ou point invalide
            elif bpi[i, j] <= -1 or bpi[i, j] >= 1:
                classe[i, j] = Terrain.RIDULE
            else:
                classe[i, j] = Terrain.LISSE
    return classe


# affichage des 2 classifications dasn le même graphique avec la classif 1 en couleur pleine et la classif 2 en hachurage
def afficher_classifications(mnt):
    mnt = mnt[::-1, ::]
    classif1 = classif_1(mnt)
    classif2 = classif_2(mnt)

    _, ax = plt.subplots(figsize=(12, 6))

    # Classif 1 en couleur de fond
    im = ax.imshow(classif1, cmap="tab10", vmin=0, vmax=5, alpha=0.8)

    # Classif 2 : hachures sur les zones DUNE
    # imshow inverts the y-axis, contourf aligns naturally in those coordinates
    dune_mask = (classif2 == Terrain.RIDULE).astype(float)
    ax.contour(dune_mask, levels=[0.5], colors="black", hatches=["///"], linewidths=0)

    # Restaurer les limites définies par imshow (contourf peut les modifier)
    h, w = classif1.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    ax.set_title("Classification des terrains")

    cmap = plt.cm.tab10
    norm = plt.Normalize(vmin=0, vmax=5)
    def c(val):
        return cmap(norm(val))

    legend_elements = [
        Patch(facecolor=c(Terrain.PLAT),       edgecolor="k", label="Plat"),
        Patch(facecolor=c(Terrain.DEPRESSION),  edgecolor="k", label="Dépression"),
        Patch(facecolor=c(Terrain.CRETE),       edgecolor="k", label="Crête"),
        Patch(facecolor=c(Terrain.PENTE),       edgecolor="k", label="Pente"),
        Patch(facecolor="none", edgecolor="k", hatch="///", label="Ridules (hachures)"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    print(f"Nombre de pixels 'Dune' trouvés : {np.sum(dune_mask)}")
    plt.tight_layout()
    plt.show()

# Chargement des MNT artificiels
def resolve(name):
    p = Path(name)
    return p if p.exists() else Path(f"./../{name}")

data1 = np.loadtxt(resolve("Dune2_Dunkerque_Extrait1_50cm.xyz"))

print(f"Affichage des classifications pour Dunkerque")
afficher_classifications(data1)
