from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
cmap = plt.cm.gist_earth
img1 = axs[0, 0].contourf(X, Y, data1, levels=100, cmap=cmap)
axs[0, 0].set_title('sincard')
# On attache la colorbar spécifiquement à cet axe
fig.colorbar(img1, ax=axs[0, 0], label='Altitude [m]')


img2 = axs[0, 1].contourf(X, Y, data2, levels=100, cmap=cmap)
axs[0, 1].set_title('plateau')
fig.colorbar(img2, ax=axs[0, 1], label='Altitude [m]')


img3 = axs[1, 0].contourf(X, Y, data3, levels=100, cmap=cmap)
axs[1, 0].set_title('plan')
fig.colorbar(img3, ax=axs[1, 0], label='Altitude [m]')


img4 = axs[1, 1].contourf(X, Y, data4, levels=100, cmap=cmap)
axs[1, 1].set_title('double_sin')
fig.colorbar(img4, ax=axs[1, 1], label='Altitude [m]')

plt.show()
