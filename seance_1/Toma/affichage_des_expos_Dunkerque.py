from pathlib import Path
from pentes_Toma import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dune_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not dune_path.exists():
    dune_path = Path("../../Dune2_Dunkerque_Extrait1_50cm.xyz")


data = np.loadtxt(dune_path)

x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)
Z = exposition(*Evans(data))
cmap = plt.cm.gist_earth
img = plt.contourf(X, Y, Z, levels=100, cmap=cmap)
# Autre possibilité
# img = plt.pcolormesh(X, Y, data, cmap=cmap)
plt.title('Dunkerque Exposition')
plt.colorbar(img, label='exposition')

plt.show()
