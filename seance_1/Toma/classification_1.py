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
    CRETE = 2
    PENTE = 3
    DUNE = 4
    PAS_DUNE = 5


def classif_1(mnt):
    bpi = calcul_BPI(mnt, r=30)
    classe = np.zeros_like(mnt)
    pts = pente(*Evans(mnt))
    # parcourir les bpi pour trier les points
    for i in range(bpi.shape[0]):
        for j in range(bpi.shape[1]):
            if np.isnan(bpi[i, j]):
                classe[i, j] = -1  # bord ou point invalide
            elif bpi[i, j] <= -1:
                classe[i, j] = Terrain.DEPRESSION
            elif bpi[i, j] >= 1:
                classe[i, j] = Terrain.CRETE
            else:
                if pts[i, j] < 0.18:
                    classe[i, j] = Terrain.PLAT
                else:
                    classe[i, j] = Terrain.PENTE
    return classe


def classif_2(mnt):
    bpi = calcul_BPI(mnt, r=7)
    classe = np.zeros_like(mnt)
    # parcourir les bpi pour trier les points
    for i in range(bpi.shape[0]):
        for j in range(bpi.shape[1]):
            if np.isnan(bpi[i, j]):
                classe[i, j] = -1  # bord ou point invalide
            elif bpi[i, j] <= -1 or bpi[i, j] >= 1:
                classe[i, j] = Terrain.DUNE
            else:
                classe[i, j] = Terrain.PAS_DUNE
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
    dune_mask = (classif2 == Terrain.DUNE).astype(float)
    ax.contourf(dune_mask, levels=[0.5, 1.5], hatches=["///"], colors="black", alpha=0)

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
        Patch(facecolor=c(Terrain.DUNE),        edgecolor="k", label="Dune"),
        Patch(facecolor=c(Terrain.PAS_DUNE),    edgecolor="k", label="Pas dune"),
        Patch(facecolor="none", edgecolor="k", hatch="///", label="Dune (hachures)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.show()

# Chargement des MNT artificiels
def resolve(name):
    p = Path(name)
    return p if p.exists() else Path(f"./../{name}")

data1 = np.loadtxt(resolve("Dune2_Dunkerque_Extrait1_50cm.xyz"))

print(f"Affichage des classifications pour Dunkerque")
afficher_classifications(data1)
