from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from recap import calcul_BPI
from pentes_Toma import pente, Evans
from rugosite2 import matrice_rugo
from matplotlib.colors import ListedColormap
import matplotlib as mpl


mpl.use("TkAgg") # pour ne pas appler Qt qui fonctionne mal avec Wayland
mpl.rcParams['figure.dpi'] = 175      # augmente la résolution d'affichage
mpl.rcParams['figure.figsize'] = (8, 6)  # taille par défaut des figures (en pouces)


class Terrain:
    PLAT = 0
    P_DOUCE = 1
    P_RAIDE = 3
    CRETE = 2
    RIDULE = 4
    LISSE = 5
    DEEPFLAT = 6
    SHALLOWFLAT = 7
    MIDFLAT = 8
    NAN = 9
    DUNE_FOOT = 10


def classif_1(mnt):
    bpi = calcul_BPI(mnt, r=35)
    # FIX : initialiser à NAN plutôt qu'à 0 pour éviter des pixels PLAT parasites
    classe = np.full_like(mnt, Terrain.NAN, dtype=float)
    pts = pente(*Evans(mnt))

    nan_mask  = np.isnan(bpi)
    # dep_mask  = ~nan_mask & (bpi <= -0.4)
    low_bpi = ~nan_mask & (bpi <= -0.4)
    high_bpi = ~nan_mask & (bpi >= 0.5)
    mid_bpi = ~nan_mask & ~low_bpi & ~high_bpi
    flat_mask  = ~nan_mask & (pts < 0.15)
    pt_raide = ~nan_mask & (pts >= 0.15)
    pt_doux = ~nan_mask & ~flat_mask & ~pt_raide

    classe[low_bpi]                                                   = Terrain.DUNE_FOOT
    classe[high_bpi]                                                  = Terrain.CRETE
    classe[mid_bpi & (pts > 0.15)]                                    = Terrain.P_RAIDE
    classe[mid_bpi & (0.15 < pts) & (pts <= 0.15)]                    = Terrain.P_DOUCE  # volontairement impossible à vérifier pour avoir une meilleure lisibilité de la carte
    classe[mid_bpi & (pts < 0.15) & (mnt <= -33)]                     = Terrain.DEEPFLAT
    classe[mid_bpi & (pts < 0.15) & (-33 < mnt) & (mnt <= -27)]       = Terrain.MIDFLAT
    classe[mid_bpi & (pts < 0.15) & (-27 < mnt)]                      = Terrain.SHALLOWFLAT
    return classe


def classif_2(mnt):
    bpi = calcul_BPI(mnt, r=15)
    # FIX : initialiser à NAN plutôt qu'à 0
    classe = np.full_like(mnt, Terrain.NAN, dtype=float)

    nan_mask   = np.isnan(bpi)
    ridule_mask = ~nan_mask & ((bpi <= -1) | (bpi >= 1))
    lisse_mask  = ~nan_mask & ~ridule_mask

    classe[ridule_mask] = Terrain.RIDULE
    classe[lisse_mask]  = Terrain.LISSE
    return classe


def afficher_classifications(mnt):
    mnt = mnt[::-1, ::]
    classif1 = classif_1(mnt)
    classif2 = classif_2(mnt)
    print("calcul de rugo en cours")
    mat_rugo = matrice_rugo()
    print("calcul rugo fini")
    mat_rugo = mat_rugo[::-1, ::]

    rugo_haute = np.where(mat_rugo > 0.006, 1, np.nan)

    _, ax = plt.subplots(figsize=(12, 6))

    # 1. Définition de la colormap personnalisée
    # Ordre : PLAT, P_DOUCE, CRETE, P_RAIDE, RIDULE, LISSE, DEEPFLAT, SHALLOWFLAT, MIDFLAT, NAN, DUNE_FOOT
    couleurs = ["white", "orange", "red", "purple", "green", "yellow", "darkblue", "lightblue", "blue", "gray", "lightgreen"]
    cmap = ListedColormap(couleurs)
    norm = plt.Normalize(vmin=0, vmax=10)

    # FIX : vmin=0, vmax=9 pour couvrir toutes les valeurs de Terrain
    ax.imshow(classif1, cmap=cmap, norm=norm, alpha=0.8)

    # Classif 2 : hachures sur les zones RIDULE (commenté)
    # dune_mask = (classif2 == Terrain.RIDULE).astype(float)
    # ax.contour(dune_mask, levels=[0.5], colors="black", hatches=["///"], linewidths=0)

    h, w = classif1.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_title("Classification des terrains")

    legend_elements = [
        Patch(facecolor=cmap(Terrain.P_DOUCE),      edgecolor="k", label="Pente Douce"),
        Patch(facecolor=cmap(Terrain.CRETE),        edgecolor="k", label="Crête"),
        Patch(facecolor=cmap(Terrain.P_RAIDE),      edgecolor="k", label="Pente raide"),
        Patch(facecolor=cmap(Terrain.DEEPFLAT),     edgecolor="k", label="Deep Flat"),
        Patch(facecolor=cmap(Terrain.MIDFLAT),      edgecolor="k", label="Mid Flat"),
        Patch(facecolor=cmap(Terrain.SHALLOWFLAT),  edgecolor="k", label="Shallow Flat"),
        Patch(facecolor=cmap(Terrain.DUNE_FOOT),      edgecolor="k", label="Pied de dune"),
        Patch(facecolor=cmap(Terrain.NAN),          edgecolor="k", label="NaN / bord"),
        Patch(facecolor="black", edgecolor="k", label="Rugosité")
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    im_rugo = ax.imshow(rugo_haute, cmap="Blues", vmin=0, vmax=1, alpha=0.5, interpolation='none', zorder=10)
    np.savetxt("test_mat.txt",classif1,fmt = "%d")
    plt.tight_layout()
    plt.show()

def resolve(name):
    p = Path(name)
    return p if p.exists() else Path(f"./../{name}")



data1 = np.load("Dune2_Dunkerque_Extrait1_50cm.npy")
print("Affichage des classifications pour Dunkerque", flush=True)
print((data1.shape))
afficher_classifications(data1)

#1237 par 1249

