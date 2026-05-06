from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

dune_path = Path("Dune2_Dunkerque_Extrait1_50cm.xyz")
if not dune_path.exists():
    dune_path = Path("../../Dune2_Dunkerque_Extrait1_50cm.xyz")


data = np.loadtxt(dune_path)

#Calcul de la moyenne
print(f"La moyenne de profondeur à Dunkerque est : {np.mean(data)}")

#Calcul du min
print(f"Le min de profondeur à Dunkerque est : {np.min(data)}")

#Calcul du max
print(f"Le max de profondeur à Dunkerque est : {np.max(data)}")

#Calcul de l'écart type
print(f"L'écart type de profondeur à Dunkerque est : {np.std(data)}")


# On décore un peu le graphique
plt.title("Histogramme bathymétrique")
plt.xlabel("Profondeur")
plt.ylabel("Occurences")

# On trace l'histogramme
plt.hist(data.flatten(), bins=100, edgecolor="black", linewidth=0.5)

# Affichage de la figure
plt.show()

