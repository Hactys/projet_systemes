from pathlib import Path
from pentes_Toma import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage

# --- 1. Gestion des chemins et chargement ---
sin_card_path = Path("sin_card.txt") if Path("sin_card.txt").exists() else Path("./../sin_card.txt")
plateau_path = Path("plateau.txt") if Path("plateau.txt").exists() else Path("./../plateau.txt")
plan_path = Path("plan.txt") if Path("plan.txt").exists() else Path("./../plan.txt")
double_sin_path = Path("double_sin.txt") if Path("double_sin.txt").exists() else Path("./../double_sin.txt")
berthaume_path = Path("bertheaume_z.txt") if Path("bertheaume_z.txt").exists() else Path("./../bertheaume_z.txt")

dunkerque_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not dunkerque_path.exists():
    dunkerque_path = Path("./../Dune2_Dunkerque_Extrait1_50cm.xyz")

data1 = np.loadtxt(sin_card_path)
data2 = np.loadtxt(plateau_path)
data3 = np.loadtxt(plan_path)
data4 = np.loadtxt(double_sin_path)
data5 = np.loadtxt(berthaume_path)
data6 = np.loadtxt(dunkerque_path)

# --- 2. Configuration des données de base ---
matrice_pente = data6
taille_filtre = 3

# --- 3. Fonction d'écart-type local (Optimisée et stricte) ---
def std_local_strict(matrice, taille):
    # Utilisation de generic_filter pour retrouver exactement ton comportement d'origine sans boucle for
    return scipy.ndimage.generic_filter(matrice, np.std, size=taille, mode='constant', cval=np.nan)

# --- 4. Affichage du premier graphique (Origine vs Filtré à 0.5) ---
filtered_matrice_init = scipy.ndimage.gaussian_filter(matrice_pente, 5.0)
matrice_residuelle = filtered_matrice_init - matrice_pente

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

im1 = ax1.imshow(std_local_strict(matrice_pente,taille_filtre), origin='lower', cmap=plt.cm.hot_r)
ax1.set_title(f"Sans filtre gaussien\n(Filtre {taille_filtre}x{taille_filtre})")
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, label='Écart-type de la pente [°]', cax=cax1)

im2 = ax2.imshow(std_local_strict(matrice_pente - filtered_matrice_init,taille_filtre), origin='lower', cmap=plt.cm.hot_r)
ax2.set_title(f"Avec filtre local\n(Filtre {taille_filtre}x{taille_filtre})")
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, label='Écart-type de la pente [°]', cax=cax2)
plt.tight_layout()



# Un seul show à la fin pour ouvrir toutes les fenêtres en même temps
plt.show()