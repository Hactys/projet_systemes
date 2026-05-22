from pathlib import Path
from pentes_Toma import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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

data1 = np.loadtxt(sin_card_path)
data2= np.loadtxt(plateau_path)
data3= np.loadtxt(plan_path)
data4 = np.loadtxt(double_sin_path)
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


cmap = plt.cm.bwr

noms_donnees = ["sin_card", "plateau", "plan", "double_sin"] #Utile pour les titres des graphiques
noms_methodes = ["TPP", "FCN", "Evans", "Théorique"] #Utile pour les titres des graphiques

'''for j, data in enumerate([data1, data2, data3, data4]):
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    theorique = [th1, th2, th3, th4][j]
    for i, mth in enumerate([TPP, FCN, Evans, theorique]):
        fx, fy = mth(data)
        exp = exposition(fx, fy)
        im = ax[i//2, i%2].imshow(exp, origin='lower', cmap=cmap)
        titre = f"{noms_donnees[j]} - Méthode : {noms_methodes[i]}"
        ax[i // 2, i % 2].set_title(titre)
        divider = make_axes_locatable(ax[i//2, i%2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='p[°]', cax=cax)
    plt.tight_layout()
    plt.show()'''


for j, data in enumerate([data1, data2, data3, data4]):
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    theorique = [th1, th2, th3, th4][j]
    exp_th=exposition(*theorique(data))
    for i, mth in enumerate([TPP, FCN, Evans, theorique]):
        print(["TPP", "FCN", "Evans", "theorique"][i])
        fx, fy = mth(data)
        exp = exposition(fx, fy)
        pmax = 0.1
        normalize = Normalize(-pmax, pmax)
        im = ax[i//2, i%2].imshow(exp-exp_th, origin='lower', cmap=cmap, norm=normalize)
        titre = f"{noms_donnees[j]} - Méthode : {noms_methodes[i]}"
        ax[i // 2, i % 2].set_title(titre)
        divider = make_axes_locatable(ax[i//2, i%2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='p[°]', cax=cax)
    plt.tight_layout()
    plt.show()