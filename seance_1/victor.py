from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg") # pour ne pas appler Qt qui fonctionne mal avec Wayland
mpl.rcParams['figure.dpi'] = 175      # augmente la résolution d’affichage
mpl.rcParams['figure.figsize'] = (8, 6)  # taille par défaut des figures (en pouces)



dune_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not dune_path.exists():
    dune_path = Path("./../Dune2_Dunkerque_Extrait1_50cm.xyz")


data = np.loadtxt(dune_path)

x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)

cmap = plt.cm.gist_earth
img = plt.contourf(X, Y, data, levels=100, cmap=cmap)
# plt.contour(X, Y, data, levels=5, colors='black')
# Autre possibilité
# img = plt.pcolormesh(X, Y, data, cmap=cmap)
plt.title('Dunkerque')
plt.colorbar(img, label='Altitude [m]')

plt.show()
