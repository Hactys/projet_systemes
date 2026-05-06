import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)
mnt = np.loadtxt('Dune2_Dunkerque_Extrait1_50cm.xyz')
mini = np.min(mnt)
maxi = np.max(mnt)
moy = np.mean(mnt)
print(f"Le max est {mini}")

print(f"Le min est {maxi}")
print(f"La moyenne est {moy}")

cmap = plt.cm.gist_earth

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, mnt, cmap=cmap, alpha=0.9)  # légère transparence

# --- Lignes de niveau superposées sur la surface ---
contours = ax.contour(X, Y, mnt, levels=15, cmap='rainbow',zdir='z',offset=None)

ax.set_title('Plan 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Altitude [m]')
fig.colorbar(surf, ax=ax, label='Altitude [m]', shrink=0.5)

plt.show()