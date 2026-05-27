from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage


dunkerque_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not dunkerque_path.exists():
    dunkerque_path = Path("./../Dune2_Dunkerque_Extrait1_50cm.xyz")

raw = np.loadtxt(dunkerque_path)

data = raw.astype(float)

# Nettoyage des valeurs nodata (typiquement -9999 ou 9999)
NODATA_THRESHOLD = -100.0
data[data < NODATA_THRESHOLD] = np.nan


# ==========================================
# 2. GRADIENTS — MÉTHODE D'EVANS
# ==========================================
def Evans(mnt, s=1.0):
    """Calcule les gradients fx et fy avec la méthode d'Evans (3×3)."""
    matx = np.full_like(mnt, np.nan, dtype=float)
    maty = np.full_like(mnt, np.nan, dtype=float)

    z1 = mnt[:-2, :-2]; z2 = mnt[:-2, 1:-1]; z3 = mnt[:-2, 2:]
    z4 = mnt[1:-1, :-2];                      z6 = mnt[1:-1, 2:]
    z7 = mnt[2:,  :-2]; z8 = mnt[2:,  1:-1]; z9 = mnt[2:,  2:]

    matx[1:-1, 1:-1] = (z3 + z6 + z9 - (z1 + z4 + z7)) / (6 * s)
    maty[1:-1, 1:-1] = -((z1 + z2 + z3 - (z7 + z8 + z9)) / (6 * s))

    return matx, maty


def exposition(fx, fy):
    """Calcule l'exposition en radians."""
    return np.arctan2(-fx, -fy)


# ==========================================
# 3. RUGOSITÉ — VERSION CORRIGÉE (NaN-aware)
# ==========================================
def _nansum_3x3(values):
    """Fonction de somme ignorant les NaN, pour generic_filter."""
    s = np.nansum(values)
    # Si tous les pixels du voisinage sont NaN → renvoyer NaN
    return np.nan if np.all(np.isnan(values)) else s


def rugosite(matrice_pente_rad, matrice_exposition_rad, taille_voisinage=3):
    """
    Calcule la rugosité d'après la dispersion des vecteurs normaux.

    Correction clé : utilisation de generic_filter + nansum au lieu de
    uniform_filter qui propage les NaN et produit une image blanche.
    """
    n = taille_voisinage ** 2

    # Composantes du vecteur normal unitaire (éq. 9)
    x = np.sin(matrice_pente_rad) * np.cos(matrice_exposition_rad)
    y = np.sin(matrice_pente_rad) * np.sin(matrice_exposition_rad)
    z = np.cos(matrice_pente_rad)

    # Somme locale NaN-aware sur le voisinage (éq. 10)
    # generic_filter est plus lent mais robuste aux NaN,
    # contrairement à uniform_filter qui les propage.
    kw = dict(size=taille_voisinage, mode="reflect")
    x_bar = scipy.ndimage.generic_filter(x, _nansum_3x3, **kw)
    y_bar = scipy.ndimage.generic_filter(y, _nansum_3x3, **kw)
    z_bar = scipy.ndimage.generic_filter(z, _nansum_3x3, **kw)

    # Norme du vecteur résultant (éq. 11)
    r = np.sqrt(x_bar**2 + y_bar**2 + z_bar**2)

    # Indice de rugosité k ∈ [0, 1] (éq. 12)
    k = np.clip(1 - (r / n), 0, 1)

    # Remise à NaN des pixels sans données valides
    masque_nan = np.isnan(matrice_pente_rad) | np.isnan(matrice_exposition_rad)
    k[masque_nan] = np.nan

    # Bord du filtre → NaN
    demi = taille_voisinage // 2
    if demi > 0:
        k[:demi, :]  = np.nan
        k[-demi:, :] = np.nan
        k[:, :demi]  = np.nan
        k[:, -demi:] = np.nan

    return k

def mainne():
    fx, fy = Evans(data)

    norme_gradient  = np.sqrt(fx**2 + fy**2)
    mat_pente_rad   = np.arctan(norme_gradient)
    mat_expo_rad    = exposition(fx, fy)
    mat_rugosite    = rugosite(mat_pente_rad, mat_expo_rad, taille_voisinage=3)

    return mat_rugosite

# matrice_rugueuse = mainne()
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# # — MNT d'origine
# ax = axes[0] # *
# im = ax.imshow(data, origin="lower", cmap="hot_r")
# ax.set_title("MNT — Altitude (m)")
# divider = make_axes_locatable(ax)
# plt.colorbar(im, label="Altitude [m]", cax=divider.append_axes("right", size="5%", pad=0.05))

# # — Rugosité
# ax = axes[1] # <- Second axe (anciennement 2)
# im = ax.imshow(matrice_rugueuse, origin="lower", cmap="hot_r", vmin=0, vmax=np.nanpercentile(matrice_rugueuse, 99))
# ax.set_title("Rugosité — Dispersion vecteurs normaux 3×3")
# divider = make_axes_locatable(ax)
# plt.colorbar(im, label="Indice de rugosité k", cax=divider.append_axes("right", size="5%", pad=0.05))

# plt.suptitle("Analyse de surface — Dunkerque (résolution 50 cm)", fontsize=13, y=1.02)
# plt.tight_layout()
# plt.show()

