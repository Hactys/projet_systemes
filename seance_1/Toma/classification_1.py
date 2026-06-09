from pathlib import Path
from tkinter import N
from typing import List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix
from skimage.filters.rank import modal

from recap import calcul_BPI
from rugosite2 import matrice_rugo
from pentes_Toma import pente, Evans



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


def k_moyenne(mnt, n_clusters: List[int], sigma: float = 2.0, smooth=False):
    PAS = 0.5
    x = np.arange(mnt.shape[1]) * PAS
    y = np.arange(mnt.shape[0]) * PAS
    X, Y = np.meshgrid(x, y)
    w_bpi = calcul_BPI(mnt, r=35)
    pts = pente(*Evans(mnt))
    z = mnt.copy()

    if smooth:
        # Lissage gaussien des features pour des clusters plus cohérents spatialement
        z = gaussian_filter(mnt, sigma=sigma)
        pts = gaussian_filter(pts, sigma=sigma)
        w_bpi = gaussian_filter(w_bpi, sigma=sigma)

    df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten()[::-1], 'z': z.flatten(), 'p': pts.flatten(), 'wbpi': w_bpi.flatten()})
    data = df.dropna().copy()
    select = data[['z', 'p', 'wbpi']]

    # Mise à l'échelle des données
    scaler = StandardScaler()
    data_ok = scaler.fit_transform(select)
    #data_ok[1] *= 10

    resultats = {}
    for n in n_clusters:
        kmeans = cluster.KMeans(n_clusters=n, random_state=0).fit(data_ok)
        df[f'kmeans_{n}'] = pd.Series(kmeans.labels_, index=data.index)
        mat = df.pivot(columns='x', index='y')
        resultats[n] = mat[f'kmeans_{n}'].values
    return resultats


def afficher_kmeans(mnt, n_clusters:List[int]):
    fig, axes = plt.subplots(len(n_clusters), 3, figsize=(15, 5))
    # boucle qui affiche chaque classification dans un subplot différent
    # première colonne : classification k-means
    # deuxième colonne : classification k-means avec lissage gaussien
    # troisième colonne : classification k-means avec lissage gaussien plus fort
    for i, n in enumerate(n_clusters):
        resultats = k_moyenne(mnt, n_clusters=[n], sigma=2.0, smooth=False)
        axes[i, 0].imshow(resultats[n], cmap='tab10')
        axes[i, 0].set_title(f'K-means (n={n})')

        resultats_smooth = k_moyenne(mnt, n_clusters=[n], sigma=1.25, smooth=True)
        axes[i, 1].imshow(resultats_smooth[n], cmap='tab10')
        axes[i, 1].set_title(f'K-means lissé (n={n})')

        resultats_smooth_strong = k_moyenne(mnt, n_clusters=[n], sigma=3.0, smooth=True)
        axes[i, 2].imshow(resultats_smooth_strong[n], cmap='tab10')
        axes[i, 2].set_title(f'K-means lissé fort (n={n})')

        # mettre legende de la forme : ax.legend(handles=[Patch(facecolor=plt.cm.viridis(i / max(n_clusters)), edgecolor="k", label=f"Cluster {i}") for i in range(k)], bbox_to_anchor=(0.98, 0.95), loc="upper left")
        axes[i, 0].legend(handles=[Patch(facecolor=plt.cm.viridis(j / n), edgecolor="k", label=f"Cluster {j}") for j in range(n)], bbox_to_anchor=(0.98, 0.95), loc="upper left")
        axes[i, 1].legend(handles=[Patch(facecolor=plt.cm.viridis(j / n), edgecolor="k", label=f"Cluster {j}") for j in range(n)], bbox_to_anchor=(0.98, 0.95), loc="upper left")
        axes[i, 2].legend(handles=[Patch(facecolor=plt.cm.viridis(j / n), edgecolor="k", label=f"Cluster {j}") for j in range(n)], bbox_to_anchor=(0.98, 0.95), loc="upper left")
    plt.tight_layout()
    plt.show()


def matrices_confusion(mnt, n_clusters:List[int]):
    terrain_names = {v: k for k, v in Terrain.__dict__.items() if not k.startswith('_')}

    classif1 = classif_1(mnt)
    resultats = k_moyenne(mnt, n_clusters=n_clusters, sigma=1.5, smooth=False)

    n_plots = len(n_clusters)
    fig, axes = plt.subplots(1, n_plots, figsize=(max(6, sum(n * 0.9 for n in n_clusters)), 6))
    if n_plots == 1:
        axes = [axes]

    # pré-calcul de toutes les matrices pour déterminer le max global
    cms_pct = []
    classif_classes_list = []
    for n in n_clusters:
        kmeans_labels = resultats[n].flatten()
        classif_labels = classif1.flatten()
        mask = ~np.isnan(classif_labels) & ~np.isnan(kmeans_labels)
        kmeans_labels = kmeans_labels[mask].astype(int)
        classif_labels = classif_labels[mask].astype(int)

        classif_classes = np.unique(classif_labels)
        cm = np.array([
            np.bincount(kmeans_labels[classif_labels == c], minlength=n)
            for c in classif_classes
        ])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0.0)
        cms_pct.append(cm_pct)
        classif_classes_list.append(classif_classes)

    vmax = max(cm.max() for cm in cms_pct)

    for ax, n, cm_pct, classif_classes in zip(axes, n_clusters, cms_pct, classif_classes_list):
        im = ax.imshow(cm_pct, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto', vmin=0, vmax=vmax)
        ax.set_title(f'k={n}')

        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([f'C{i}' for i in range(n)], rotation=45, ha='right')
        ax.set_xlabel(f'Cluster {n}-means')

        ax.set_yticks(np.arange(len(classif_classes)))
        ax.set_yticklabels([terrain_names.get(c, str(c)) for c in classif_classes])

        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                ax.text(j, i, f'{cm_pct[i, j]:.1f}%', ha='center', va='center',
                        color='white' if cm_pct[i, j] > vmax / 2 else 'black', fontsize=7)

    fig.colorbar(im, ax=axes, label='%', pad=-0.45)
    fig.suptitle('Matrices de confusion — classif_1 vs k-means (% par classe)', y=1.01)
    plt.tight_layout()
    plt.show()

data1 = np.load("Dune2_Dunkerque_Extrait1_50cm.npy")
print("Affichage des classifications pour Dunkerque", flush=True)
print((data1.shape))
# afficher_classifications(data1)
#afficher_kmeans(data1, n_clusters=[5, 6])
matrices_confusion(data1, n_clusters=[5, 6])

#1237 par 1249

