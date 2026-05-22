from pathlib import Path
from pentes_Toma import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sin_card_path = Path("sin_card.txt")
if not sin_card_path.exists():
    sin_card_path = Path("./../sin_card.txt")

plateau_path = Path("plateau.txt")
if not plateau_path.exists():
    plateau_path = Path("./../plateau.txt")

plan_path = Path("plan.txt")
if not plan_path.exists():
    plan_path = Path("./../plan.txt")

double_sin_path = Path("double_sin.txt")
if not double_sin_path.exists():
    double_sin_path = Path("./../double_sin.txt")

berthaume_path = Path("bertheaume_z.txt")
if not berthaume_path.exists():
    berthaume_path = Path("./../bertheaume_z.txt")

dunkerque_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not berthaume_path.exists():
    berthaume_path = Path("./../Dune2_Dunkerque_Extrait1_50cm.xyz")

data1 = np.loadtxt(sin_card_path)
data2= np.loadtxt(plateau_path)
data3= np.loadtxt(plan_path)
data4 = np.loadtxt(double_sin_path)
data5 = np.loadtxt(berthaume_path)
data6 = np.loadtxt(dunkerque_path)

x = np.arange(0, 101)
y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)

d = np.sqrt((X - 40)**2 + (Y - 50)**2)
# Sécurité : on remplace temporairement les 0 par 1 pour éviter
# le crash de la division par zéro au point (x=40, y=50)
d_safe = np.where(d == 0, 1, d)
num1 = 10 * d_safe * np.cos(0.1 * d_safe) - 100 * np.sin(0.1 * d_safe)
th1 = lambda _ : (np.where(d == 0, 0, (X - 40) * num1 / (d_safe**3)), np.where(d == 0, 0, (Y - 50) * num1 / (d_safe**3)))
th2 = lambda _ : (1 - np.tanh((X - 40) / 5)**2, np.zeros_like(Y))
th3 = lambda _ : (np.full(X.shape, 0.07), np.full(Y.shape, 0.1))
th4 = lambda _ : (0.5 * np.cos((X / 10) + 3 * np.sin(Y / 20)), 0.75 * np.cos(Y / 20) * np.cos((X / 10) + 3 * np.sin(Y / 20)) + 0.4 * np.cos(Y / 5))


cmap = plt.cm.gist_earth_r
fx_th1, fy_th1 = th1(None)
mat1 = pente(fx_th1, fy_th1)
print(f"matrice 1 {mat1}")
taille_filtre = 5

def extraire_fenetre_centree_strict(coord, matrice, taille):
    l_centre, c_centre = coord
    demi_taille = taille // 2
    H, L = matrice.shape
    
    l_min = l_centre - demi_taille
    l_max = l_centre + demi_taille + 1
    c_min = c_centre - demi_taille
    c_max = c_centre + demi_taille + 1
    
    if l_min < 0 or l_max > H or c_min < 0 or c_max > L:
        return np.full((taille, taille), np.nan)
    
    return matrice[l_min:l_max, c_min:c_max]

fx_th2, fy_th2 = th4(None)
# matrice_pente = pente(fx_th2, fy_th2)
matrice_pente = data6


H, L = matrice_pente.shape
matrice_std = np.zeros((H, L))

for l in range(H):
    for c in range(L):
        fenetre = extraire_fenetre_centree_strict((l, c), matrice_pente, taille_filtre)
        
        if np.isnan(fenetre).any():
            matrice_std[l, c] = np.nan
        else:
            matrice_std[l, c] = np.std(fenetre)

# 6. Affichage de la matrice d'écart-type
fig, ax = plt.subplots(figsize=(8, 6))

# On utilise une palette séquentielle comme 'viridis' ou 'magma' pour l'écart-type
im = ax.imshow(matrice_std, origin='lower', cmap=plt.cm.viridis)
ax.set_title(f"Écart-type local (Filtre {taille_filtre}x{taille_filtre})")

# Ajout de la colorbar propre
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, label='Écart-type de la pente [°]', cax=cax)

plt.tight_layout()
plt.show()