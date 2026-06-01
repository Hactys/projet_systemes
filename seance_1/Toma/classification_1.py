from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from recap import calcul_BPI
from pentes_Toma import pente, Evans
from rugosite2 import matrice_rugo


class Terrain:
    PLAT = 0
    DEPRESSION = 1
    CRETE = 3
    PENTE = 2
    RIDULE = 4
    LISSE = 5
    DEEPFLAT = 6
    SHALLOWFLAT = 7
    MIDFLAT = 8
    NAN = 9


def classif_1(mnt):
    bpi = calcul_BPI(mnt, r=40)
    # FIX : initialiser à NAN plutôt qu'à 0 pour éviter des pixels PLAT parasites
    classe = np.full_like(mnt, Terrain.NAN, dtype=float)
    pts = pente(*Evans(mnt))

    nan_mask  = np.isnan(bpi)
    dep_mask  = ~nan_mask & (bpi <= -0.4)
    pente_mask = ~nan_mask & (bpi >= 0.5)
    flat_mask  = ~nan_mask & ~dep_mask & ~pente_mask & (pts < 0.18)
    crete_mask = ~nan_mask & ~dep_mask & ~pente_mask & (pts >= 0.18)

    classe[dep_mask]                            = Terrain.DEPRESSION
    classe[pente_mask]                          = Terrain.PENTE
    classe[crete_mask]                          = Terrain.CRETE
    classe[flat_mask & (mnt < -33)]             = Terrain.DEEPFLAT
    classe[flat_mask & (mnt > -27)]             = Terrain.SHALLOWFLAT
    classe[flat_mask & (mnt >= -33) & (mnt <= -27)] = Terrain.MIDFLAT
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
    # classif2 = classif_2(mnt)
    print("calcul de rugo en cours")
    mat_rugo = matrice_rugo()
    print("calcul rugo fini")
    mat_rugo = mat_rugo[::-1, ::]

    rugo_haute = np.where(mat_rugo > 0.006, 1, np.nan)

    _, ax = plt.subplots(figsize=(12, 6))

    # FIX : vmin=0, vmax=9 pour couvrir toutes les valeurs de Terrain
    cmap = plt.cm.tab10
    norm = plt.Normalize(vmin=0, vmax=9)

    ax.imshow(classif1, cmap=cmap, norm=norm, alpha=0.8)

    # Classif 2 : hachures sur les zones RIDULE (commenté)
    # dune_mask = (classif2 == Terrain.RIDULE).astype(float)
    # ax.contour(dune_mask, levels=[0.5], colors="black", hatches=["///"], linewidths=0)

    h, w = classif1.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_title("Classification des terrains")

    def c(val):
        return cmap(norm(val))

    legend_elements = [
        Patch(facecolor=c(Terrain.DEPRESSION),  edgecolor="k", label="Versant abrupte"),
        Patch(facecolor=c(Terrain.PENTE),        edgecolor="k", label="Versant doux"),
        Patch(facecolor=c(Terrain.CRETE),        edgecolor="k", label="Crête"),
        Patch(facecolor=c(Terrain.DEEPFLAT),     edgecolor="k", label="Deep Flat"),
        Patch(facecolor=c(Terrain.MIDFLAT),      edgecolor="k", label="Mid Flat"),
        Patch(facecolor=c(Terrain.SHALLOWFLAT),  edgecolor="k", label="Shallow Flat"),
        Patch(facecolor=c(Terrain.NAN),          edgecolor="k", label="NaN / bord"),
        Patch(facecolor="none", edgecolor="k", hatch="///", label="Ridules (hachures)"),
        Patch(facecolor="red", edgecolor="k", label="Rugosité")
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    im_rugo = ax.imshow(rugo_haute, cmap="Reds", vmin=0, vmax=1, alpha=0.5, interpolation='none', zorder=10)
    np.savetxt("test_mat.txt",)
    plt.tight_layout()
    plt.show()


def resolve(name):
    p = Path(name)
    return p if p.exists() else Path(f"./../{name}")


data1 = np.loadtxt(resolve("Dune2_Dunkerque_Extrait1_50cm.xyz"))
print("Affichage des classifications pour Dunkerque")
afficher_classifications(data1)