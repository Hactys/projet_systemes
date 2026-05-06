from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource
import numpy as np
import matplotlib.pyplot as plt

dune_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not dune_path.exists():
    dune_path = Path("../../Dune2_Dunkerque_Extrait1_50cm.xyz")


data = np.loadtxt(dune_path)
'''arm=data

cmap = plt.cm.cubehelix

# Dimensions de l'image à afficher
x_arm = np.arange(arm.shape[1])
y_arm = np.arange(arm.shape[0])
X_ARM, Y_ARM = np.meshgrid(x_arm, y_arm)

# Normaliser les z pour définir la palette
norm = Normalize(vmin=np.nanmin(arm), vmax=np.nanmax(arm))
my_col = cmap(norm(arm))
# Illumination pour le modèle
ls = LightSource(azdeg=-45, altdeg=35)
rgb = ls.shade(arm, cmap=cmap, vert_exag=4, blend_mode='soft')

# Figure 3D
fig, axe = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
# Choix du point de vue
axe.view_init(elev=35., azim=35)
# Afficher la surface avec illumination
# Augmenter les valeurs rstride et cstride pour accélérer l'affichage
surf = axe.plot_surface(X_ARM, Y_ARM, arm, facecolors = rgb, linewidth=0, antialiased=False, rstride=3, cstride=3)
m = cm.ScalarMappable(cmap=cmap, norm=norm)

plt.colorbar(m, ax=axe, shrink=.8)
plt.tight_layout()

plt.show()'''

x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
cmap = plt.cm.gist_earth
x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)

z_min = np.min(data)
cset = ax.contour(X, Y, data, zdir='z',offset=z_min, levels=5, colors='black')
img = ax.plot_surface(X, Y, data, edgecolor='none', antialiased=False, cmap=cmap)
# Autre possibilité
#img = plt.pcolormesh(X, Y, data, cmap=cmap)
plt.title('Dunkerque')
plt.colorbar(img, label='Altitude [m]')


plt.show()

