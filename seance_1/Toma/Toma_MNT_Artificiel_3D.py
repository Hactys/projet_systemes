from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pentes_Toma import Evans, pente, exposition
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


#On affiche l'intégralité des MNT en 3D
'''sin_card_path = Path("sin_card.txt")
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

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), subplot_kw={'projection': '3d'})
cmap = plt.cm.gist_earth
img1 = axs[0, 0].plot_surface(X, Y, data1, edgecolor='none', antialiased=False, cmap=cmap)
axs[0, 0].set_title('sincard')
# On attache la colorbar spécifiquement à cet axe
fig.colorbar(img1, ax=axs[0, 0], label='Altitude [m]')


img2 = axs[0, 1].plot_surface(X, Y, data2, edgecolor='none', antialiased=False, cmap=cmap)
axs[0, 1].set_title('plateau')
fig.colorbar(img2, ax=axs[0, 1], label='Altitude [m]')


img3 = axs[1, 0].plot_surface(X, Y, data3, edgecolor='none', antialiased=False, cmap=cmap)
axs[1, 0].set_title('plan')
fig.colorbar(img3, ax=axs[1, 0], label='Altitude [m]')


img4 = axs[1, 1].plot_surface(X, Y, data4, edgecolor='none', antialiased=False, cmap=cmap)
axs[1, 1].set_title('double_sin')
fig.colorbar(img4, ax=axs[1, 1], label='Altitude [m]')

plt.show()'''







#Deuxième partie, on projette en 3D les valeurs 2D d'exposition et de pentes
# 1. Préparation des données
data = np.loadtxt("double_sin.txt")
x = np.arange(0, data.shape[1])
y = np.arange(0, data.shape[0])
X, Y = np.meshgrid(x, y)
Z = data

# 2. Calcul avec la méthode d'Evans (fenêtre 3x3)
fx, fy = Evans(Z)
p = pente(fx, fy)
exp = exposition(fx, fy)

# 3. Configuration de la figure 3D
fig = plt.figure(figsize=(16, 7))

# --- PLACAGE DE LA PENTE ---
ax1 = fig.add_subplot(121, projection='3d')
# Normalisation des valeurs pour la colormap (on ignore les NaN des bords)
norm_p = Normalize(vmin=np.nanmin(p), vmax=np.nanmax(p))
colors_p = plt.cm.seismic(p)

# Tracé de la surface avec les couleurs plaquées
surf1 = ax1.plot_surface(X, Y, Z, facecolors=colors_p, rstride=1, cstride=1, shade=False, norm=norm_p)
ax1.set_title("MNT 3D - Double sin - Pentes (Evans)")
ax1.set_zlabel("Élévation (Z)")

# Ajout d'une barre de couleur (nécessite un objet mappable artificiel)
mappable_p = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=norm_p)
mappable_p.set_array([])
fig.colorbar(mappable_p, ax=ax1, shrink=0.5, aspect=10, label="Pente")


# --- PLACAGE DE L'EXPOSITION ---
ax2 = fig.add_subplot(122, projection='3d')
norm_exp = Normalize(vmin=np.nanmin(exp), vmax=np.nanmax(exp))
# Création des couleurs (hsv est idéal pour les angles car circulaire : le début et la fin ont la même couleur)
colors_exp = plt.cm.seismic(exp)

surf2 = ax2.plot_surface(X, Y, Z, facecolors=colors_exp, rstride=1, cstride=1, shade=False, norm=norm_exp)
ax2.set_title("MNT 3D - Double sin - Expositions (Evans)")
ax2.set_zlabel("Élévation (Z)")

mappable_exp = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=norm_exp)
mappable_exp.set_array([])
fig.colorbar(mappable_exp, ax=ax2, shrink=0.5, aspect=10, label="Exposition (Radian)")

plt.tight_layout()
plt.show()