from pathlib import Path
from pentes_Toma import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Paramètre : valeurs de sigma à tester
SIGMAS = [0.01, 0.05, 0.1, 0.5]

# Chargement des MNT artificiels
def resolve(name):
    p = Path(name)
    return p if p.exists() else Path(f"./../{name}")

data1 = np.loadtxt(resolve("sin_card.txt"))
data2 = np.loadtxt(resolve("plateau.txt"))
data3 = np.loadtxt(resolve("plan.txt"))
data4 = np.loadtxt(resolve("double_sin.txt"))

x = np.arange(0, 101)
y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)

d = np.sqrt((X - 40)**2 + (Y - 50)**2)
d_safe = np.where(d == 0, 1, d)
num1 = 10 * d_safe * np.cos(0.1 * d_safe) - 100 * np.sin(0.1 * d_safe)

th1 = lambda _: (np.where(d == 0, 0, (X - 40) * num1 / d_safe**3),
                 np.where(d == 0, 0, (Y - 50) * num1 / d_safe**3))
th2 = lambda _: (1 - np.tanh((X - 40) / 5)**2, np.zeros_like(Y))
th3 = lambda _: (np.full(X.shape, 0.07), np.full(Y.shape, 0.1))
th4 = lambda _: (0.5 * np.cos(X / 10 + 3 * np.sin(Y / 20)),
                 0.75 * np.cos(Y / 20) * np.cos(X / 10 + 3 * np.sin(Y / 20)) + 0.4 * np.cos(Y / 5))

datasets   = [data1, data2, data3, data4]
theoriques = [th1, th2, th3, th4]
noms_donnees  = ["sin_card", "plateau", "plan", "double_sin"]
noms_methodes = ["TPP", "FCN", "Evans", "Théorique"]

cmap = plt.cm.bwr

def erreur_angulaire(exp, exp_th):
    diff = exp - exp_th
    return np.arctan2(np.sin(diff), np.cos(diff))


N_BRUITS = 100
rng = np.random.default_rng(seed=42)

for sigma in SIGMAS:
    for j, (data, th) in enumerate(zip(datasets, theoriques)):

        theorique = th
        H, W = data.shape

        # Accumulation des pentes et expositions sur N_BRUITS réalisations
        pentes_stack = {nom: np.zeros((N_BRUITS, H, W)) for nom in noms_methodes}
        expos_stack  = {nom: np.zeros((N_BRUITS, H, W)) for nom in noms_methodes}

        for k in range(N_BRUITS):
            bruit = rng.normal(0, sigma, data.shape)
            data_bruite = data + bruit
            for nom, mth in zip(noms_methodes, [TPP, FCN, Evans, theorique]):
                fx, fy = mth(data_bruite)
                pentes_stack[nom][k] = pente(fx, fy)
                expos_stack[nom][k]  = exposition(fx, fy)

        # Écart-type des pentes
        fig, ax = plt.subplots(2, 2, figsize=(15, 5))
        fig.suptitle(
            f"Écart-type de la pente ({N_BRUITS} bruits) | MNT : {noms_donnees[j]} | σ = {sigma}",
            fontsize=13
        )
        for i, nom in enumerate(noms_methodes):
            std_map = np.std(pentes_stack[nom], axis=0)
            std_total = np.nanstd(std_map)
            im = ax[i // 2, i % 2].imshow(std_map, origin="lower", cmap="hot")
            ax[i // 2, i % 2].set_title(f"{noms_donnees[j]} – {nom} | σ_tot = {std_total:.4f}")
            divider = make_axes_locatable(ax[i // 2, i % 2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, label="σ(pente)", cax=cax)
        plt.tight_layout()
        plt.show()

        # Écart-type des expositions (erreur angulaire circulaire)
        fig, ax = plt.subplots(2, 2, figsize=(15, 5))
        fig.suptitle(
            f"Écart-type de l'exposition ({N_BRUITS} bruits) | MNT : {noms_donnees[j]} | σ = {sigma}",
            fontsize=13
        )
        for i, nom in enumerate(noms_methodes):
            # Écart-type circulaire : sqrt(-2 * ln(|mean(exp(i*theta))|))
            angles = expos_stack[nom]
            R = np.abs(np.mean(np.exp(1j * angles), axis=0))
            std_circ = np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1)))
            std_total = np.nanmean(std_circ)
            im = ax[i // 2, i % 2].imshow(std_circ, origin="lower", cmap="hot")
            ax[i // 2, i % 2].set_title(f"{noms_donnees[j]} : {nom} | σ_tot = {std_total:.4f} rad")
            divider = make_axes_locatable(ax[i // 2, i % 2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, label="σ circ.(exposition) [rad]", cax=cax)
        plt.tight_layout()
        plt.show()
