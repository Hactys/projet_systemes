import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)
mnt = np.loadtxt('Dune2_Dunkerque_Extrait1_50cm.xyz')

cmap = plt.cm.gist_earth

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, mnt, cmap=cmap)

ax.set_title('Plan 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Altitude [m]')
fig.colorbar(surf, ax=ax, label='Altitude [m]', shrink=0.5)

plt.show()