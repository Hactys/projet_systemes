import numpy as np
import matplotlib.pyplot as plt


#2D
# Dimensions des terrains artificiels
x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)

mnt = np.loadtxt('Dune2_Dunkerque_Extrait1_50cm.xyz')
cmap = plt.cm.gist_earth_r
img = plt.contourf(X, Y, mnt, levels=100, cmap=cmap)

plt.title('DuneDunkerque')
plt.colorbar(img,label='Altitude [m]')
print(f"ici : {mnt.shape}")
'''
#3D
#dimensions
x = np.arange(0, 1249)
y = np.arange(0, 1237)
X, Y = np.meshgrid(x, y)

mnt = np.loadtxt('Dune2Dunkerque.xyz')
cmap = plt.cm.gist_earth

# --- PASSAGE EN 3D ---
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d') # La ligne magique
plt.contour(X, Y, mnt, levels=5, colors='black')

# Tracé de la surface
img = ax.plot_surface(X, Y, mnt, cmap=cmap, edgecolor='none', antialiased=True)

ax.set_title('DuneDunkerque 3D')
ax.set_zlabel('Altitude [m]')
fig.colorbar(img, ax=ax, label='Altitude [m]', shrink=0.5)

plt.show()'''



# Calcul des statistiques
prof_min = np.min(mnt)
prof_max = np.max(mnt)
prof_moy = np.mean(mnt)
prof_std = np.std(mnt)

print(f"Caractéristiques de Dune2Dunkerque :")
print(f"— Profondeur minimale : {prof_min:.2f} m")
print(f"— Profondeur maximale : {prof_max:.2f} m")
print(f"— Profondeur moyenne  : {prof_moy:.2f} m")
print(f"— Écart-type         : {prof_std:.2f} m")


#Histogramme
#axe horizontal représente les valeurs d'altitude
#axe vertical représente la fréquence, c'est-à-dire combien de points possèdent cette altitude spécifique.
plt.figure(figsize=(10, 6))
# bins=50 définit le nombre de barres
plt.hist(mnt.flatten(), bins=50, color='skyblue', edgecolor='black')

plt.title('Histogramme des profondeurs - Dune2Dunkerque')
plt.xlabel('Altitude [m]')
plt.ylabel('Nombre de points')
plt.grid(axis='y', alpha=0.75)
print()
plt.show()