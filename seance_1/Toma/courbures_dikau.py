import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
from matplotlib.colors import ListedColormap


def coeffs_evans(mnt, s=1.0):
    z1 = mnt[:-2, :-2]
    z2 = mnt[:-2, 1:-1]
    z3 = mnt[:-2, 2:]
    z4 = mnt[1:-1, :-2]
    z5 = mnt[1:-1, 1:-1]
    z6 = mnt[1:-1, 2:]
    z7 = mnt[2:, :-2]
    z8 = mnt[2:, 1:-1]
    z9 = mnt[2:, 2:]

    A = (
        (z1 + z3 + z4 + z6 + z7 + z9) / (6 * s**2)
        - (z2 + z5 + z8) / (3 * s**2)
    )
    B = (
        (z1 + z2 + z3 + z7 + z8 + z9) / (6 * s**2)
        - (z4 + z5 + z6) / (3 * s**2)
    )
    C = (z3 + z7 - z1 - z9) / (4 * s**2)
    D = (z3 + z6 + z9 - z1 - z4 - z7) / (6 * s**2)
    E = -(z1 + z2 + z3 - z7 - z8 - z9) / (6 * s**2)
    return A, B, C, D, E


def courbures_verticale_horizontale(mnt, s=1.0):
    A, B, C, D, E = coeffs_evans(mnt, s)
    fx = D
    fy = E
    fxx = 2 * A
    fyy = 2 * B
    fxy = C
    p = fx**2 + fy**2
    q = p + 1
    kv = np.full_like(fx, np.nan)
    kh = np.full_like(fx, np.nan)
    masque = p > 1e-12
    kv[masque] = -(
        fxx[masque] * fx[masque]**2
        + 2 * fxy[masque] * fx[masque] * fy[masque]
        + fyy[masque] * fy[masque]**2
    ) / (p[masque] * np.sqrt(q[masque]**3))
    kh[masque] = -(
        fxx[masque] * fy[masque]**2
        - 2 * fxy[masque] * fx[masque] * fy[masque]
        + fyy[masque] * fx[masque]**2
    ) / (p[masque] * np.sqrt(q[masque]))
    return kv, kh


def signe_courbure(val, seuil=0.001):
    res = np.zeros_like(val, dtype=int)
    res[val > seuil] = 1
    res[val < -seuil] = -1
    return res


def classification_dikau(kv, kh, seuil=0.001):
    sv = signe_courbure(kv, seuil)
    sh = signe_courbure(kh, seuil)
    classes = np.zeros_like(kv, dtype=int)

    # ligne / colonne :
    # -1 concave
    #  0 droite
    # +1 convexe

    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
            v = sv[i, j]
            h = sh[i, j]
            if np.isnan(kv[i, j]) or np.isnan(kh[i, j]):
                classes[i, j] = -1
            elif v == 1 and h == 1:
                classes[i, j] = 0   # sommet
            elif v == 1 and h == 0:
                classes[i, j] = 1   # crête
            elif v == 1 and h == -1:
                classes[i, j] = 2   # passe
            elif v == 0 and h == 1:
                classes[i, j] = 3   # épaulement
            elif v == 0 and h == 0:
                classes[i, j] = 4   # plan
            elif v == 0 and h == -1:
                classes[i, j] = 5   # chenal
            elif v == -1 and h == 1:
                classes[i, j] = 6   # pied
            elif v == -1 and h == 0:
                classes[i, j] = 7   # vallée
            elif v == -1 and h == -1:
                classes[i, j] = 8   # cuvette
    return classes


# Affichage
if __name__ == "__main__":
    path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
    mnt = np.loadtxt(path)

    # IMPORTANT :
    # léger lissage sinon énormément de bruit
    mnt_lisse = gaussian_filter(mnt, sigma=1.2)

    kv, kh = courbures_verticale_horizontale(mnt_lisse)

    classes = classification_dikau(kv, kh, seuil=0.0005)

    labels = [
        "Sommet",
        "Crête",
        "Passe",
        "Épaulement",
        "Plan",
        "Chenal",
        "Pied",
        "Vallée",
        "Cuvette"
    ]

    cmap = ListedColormap([
        "red",
        "orange",
        "yellow",
        "lime",
        "lightgray",
        "cyan",
        "blue",
        "purple",
        "black"
    ])

    plt.figure(figsize=(10, 8))

    im = plt.imshow(classes, origin="lower", cmap=cmap)

    cbar = plt.colorbar(im)
    cbar.set_ticks(range(9))
    cbar.set_ticklabels(labels)

    plt.title("Classification de Dikau")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
