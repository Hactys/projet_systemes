from pathlib import Path
from pentes_Toma import TPP, FCN, Evans, pente
import numpy as np
import matplotlib.pyplot as plt

# Chargement MNT artificiels
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

datasets  = [data1, data2, data3, data4]
theoriques = [th1, th2, th3, th4]
noms_mnt  = ["sin_card", "plateau", "plan", "double_sin"]
methodes  = [("TPP", TPP), ("FCN", FCN), ("Evans", Evans)]

# Histogrammes des écarts : MNT artificiels
# Une figure par MNT, une colonne par méthode numérique (sans Théorique)

for j, (data, th) in enumerate(zip(datasets, theoriques)):
    pente_th = pente(*th(data))
    p_th_flat = pente_th[~np.isnan(pente_th)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Histogrammes des écarts de pente – MNT : {noms_mnt[j]}", fontsize=13)

    for ax, (nom, mth) in zip(axes, methodes):
        p = pente(*mth(data))
        ecart = (p - pente_th).flatten()
        ecart = ecart[~np.isnan(ecart)]

        ax.hist(ecart, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zéro")
        ax.axvline(np.mean(ecart), color="orange", linewidth=1.2,
                   linestyle="-", label=f"Moyenne : {np.mean(ecart):.4f}")
        ax.set_title(f"Méthode : {nom}")
        ax.set_xlabel("Écart de pente (calculée − théorique)")
        ax.set_ylabel("Occurrences")
        ax.legend(fontsize=8)

        stats = (f"μ = {np.mean(ecart):.4f}\n"
                 f"σ = {np.std(ecart):.4f}\n"
                 f"min = {np.min(ecart):.4f}\n"
                 f"max = {np.max(ecart):.4f}")
        ax.text(0.97, 0.97, stats, transform=ax.transAxes,
                fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.show()

# Histogrammes : Données bathymétriques Dunkerque
dune_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not dune_path.exists():
    dune_path = Path("../../Dune2_Dunkerque_Extrait1_50cm.xyz")

dune = np.loadtxt(dune_path)

# Histogramme des pentes calculées par chaque méthode
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Pentes calculées sur le relevé bathymétrique de Dunkerque", fontsize=13)

pentes_dune = {}
for ax, (nom, mth) in zip(axes, methodes):
    p = pente(*mth(dune)).flatten()
    p = p[~np.isnan(p)]
    pentes_dune[nom] = p

    ax.hist(p, bins=80, color="teal", edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(p), color="orange", linewidth=1.2,
               linestyle="-", label=f"Moyenne : {np.mean(p):.4f}")
    ax.set_title(f"Méthode : {nom}")
    ax.set_xlabel("Pente (sans unité)")
    ax.set_ylabel("Occurrences")
    ax.legend(fontsize=8)

    stats = (f"μ = {np.mean(p):.4f}\n"
             f"σ = {np.std(p):.4f}\n"
             f"min = {np.min(p):.4f}\n"
             f"max = {np.max(p):.4f}")
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.show()
