from pathlib import Path
from pentes_Toma import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
th1 = lambda _ : (np.cos(X), np.sin(Y))
th2 = lambda _ : (np.cos(X), np.sin(Y))
th3 = lambda _ : (np.cos(X), np.sin(Y))
th4 = lambda _ : (0.5 * np.cos((X / 10) + 3 * np.sin(Y / 20)), 0.75 * np.cos(Y / 20) * np.cos((X / 10) + 3 * np.sin(Y / 20)) + 0.4 * np.cos(Y / 5))


cmap = plt.cm.gist_earth

for j, data in enumerate([data1, data2, data3, data4]):
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    for i, mth in enumerate([TPP, FCN, Evans, [th1, th2, th3, th4][j]]):
        print(mth)
        fx, fy = mth(data)
        pt = pente(fx, fy)
        im = ax[i//2, i%2].imshow(pt, origin='lower', cmap=cmap)
        ax[i//2, i%2].set_title(["sin_card", "plateau", "plan", "double_sin"][j])
        divider = make_axes_locatable(ax[i//2, i%2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='p[°]', cax=cax)
    plt.tight_layout()
    plt.show()