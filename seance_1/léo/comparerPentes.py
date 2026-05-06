from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pentes as p
import numpy as np

# Repartir avec le terrain double sinus
mnt = np.loadtxt('double_sin.txt')
x = np.arange(0, 101)
y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)

fx, fy = 0.5 * np.cos(X/10 + 3*np.sin(Y/20)),5 * np.cos(X/10 + 3*np.sin(Y/20)) * (3/20 * np.cos(Y/20)) + 0.4 * np.cos(Y/5)
pente_reel = p.pente(fx, fy)

fx, fy = p.TPP(mnt)
pente_tpp =p.pente(fx, fy)
fx, fy = p.FCN(mnt)
pente_fcn = p.pente(fx, fy)

fx, fy = p.Evans(mnt)
pente_evans = p.pente(fx, fy)

# Normaliser les palettes entre 0° et pmax
pmax = np.max(pente_tpp)
normalize = Normalize(0, pmax)
cmap =  'cividis_r'

fig, ax = plt.subplots(1, 4, figsize=(12, 5))

im = ax[0].imshow(pente_tpp, origin='lower', cmap=cmap, norm=normalize)
ax[0].set_title('TPP')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, label='Pente[°]', cax=cax)

im = ax[1].imshow(pente_fcn, origin='lower', cmap=cmap, norm=normalize)
ax[1].set_title('FCN')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, label='Pente[°]', cax=cax)

im = ax[2].imshow(pente_evans, origin='lower', cmap=cmap, norm=normalize)
ax[2].set_title('Evans')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, label='Pente[°]', cax=cax)

im = ax[3].imshow(pente_reel, origin='lower', cmap=cmap, norm=normalize)
ax[3].set_title('Réel')
divider = make_axes_locatable(ax[3])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, label='Pente[°]', cax=cax)

plt.tight_layout()