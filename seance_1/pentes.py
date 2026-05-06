import math
import numpy as np

def TPP(mnt,dx=1, dy=1):
    """
    Calcule le gradient selon la méthode TPP (Three Points Plane)
    selon les équations :
    fx = (z3 - z2) / dx
    fy = (z1 - z2) / dy
    dx et dy représente la distance réelle dans la vraie vie entre 2 points
    """
    # z2 est le point central (pivot)
    # Pour fx, z3 est le voisin à droite (i+1)
    # Pour fy, z1 est le voisin au dessus (j+1)
    
    # fx = (z3 - z2) / delta_x
    # mnt[:, 1:] toute la matrice sauf colonne 1
    # mnt[:, :-1] toute matrice sauf derniere colonne
    # mnt[:, 1:] - mnt[:, :-1] = z3-z2 pour toute case
    fx = (mnt[:, 1:] - mnt[:, :-1]) / dx
    
    # fy = (z1 - z2) / delta_y
    # On prend toutes les colonnes, et on compare la ligne j+1 à la ligne j
    fy = (mnt[1:, :] - mnt[:-1, :]) / dy
    
    # Pour que fx et fy aient la même dimension (pour calculer la pente ensuite),
    # on réduit les matrices à leur zone commune (suppression des bords calculés)
    fx_final = fx[1:, :] # Retire la première ligne pour s'aligner sur fy
    fy_final = fy[:, 1:] # Retire la première colonne pour s'aligner sur fx
    
    return fx_final, fy_final


def FCN(mnt, dx=1, dy=1):
    """
    Calcule le gradient selon la méthode FCN (4 voisins)
    fx = (z3 - z2) / (2 * delta_x)
    fy = (z1 - z4) / (2 * delta_y)
    """
    fx = (mnt[:, 2:] - mnt[:, :-2]) / (2 * dx)
    fy = (mnt[2:, :] - mnt[:-2, :]) / (2 * dy)
    
    fx_final = fx[1:-1, :] 
    fy_final = fy[:, 1:-1]
    
    return fx_final, fy_final
    
def Evans(mnt, s=1.0):
    # Extraction des 9 voisins sur tout le MNT (sauf les bords)
    # On utilise le voisinage 3x3 de la Figure 9 du sujet
    z1 = mnt[:-2, :-2]
    z2 = mnt[:-2, 1:-1]
    z3 = mnt[:-2, 2:]
    z4 = mnt[1:-1, :-2] 
    z5 = mnt[1:-1, 1:-1]
    z6 = mnt[1:-1, 2:]
    z7 = mnt[2:, :-2] 
    z8 = mnt[2:, 1:-1] 
    z9 = mnt[2:, 2:]
    
    #en dérivant et en prenant en (0,0)
    #fx=D
    fx = (z3 + z6 + z9 - (z1 + z4 + z7)) / (6 * s**2)
    #fy=E
    fy = (z1 + z2 + z3 - (z7 + z8 + z9)) / (6 * s**2)
    
    return fx, fy

# Pour obtenir les autres coefficients (A, B, C, F) nécessaires aux courbures :
# A = (z1 + z3 + z4 + z6 + z7 + z9) / (6*s**2) - (z2 + z5 + z8) / (3*s**2)
# B = (z1 + z2 + z3 + z7 + z8 + z9) / (6*s**2) - (z4 + z5 + z6) / (3*s**2)
# C = (z3 + z7 - z1 - z9) / (4*s**2)
# F = (5*z5 + 2*(z2 + z4 + z6 + z8) - (z1 + z3 + z7 + z9)) / 9

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    