from enum import Enum

import numpy as np

from BPI import calcul_BPI
from pentes_Toma import pente, Evans


class Terrain:
    PLAT = 0
    DEPRESSION = 1
    CRETE = 2
    PENTE = 3
    DUNE = 4
    PAS_DUNE = 5


def classif_1(mnt):
    bpi = calcul_BPI(mnt, r=6)
    classe = np.zeros_like(mnt)
    pts = pente(*Evans(mnt))
    # parcourir les bpi pour trier les points
    for i in range(bpi.shape[0]):
        for j in range(bpi.shape[1]):
            if np.isnan(bpi[i, j]):
                classe[i, j] = -1  # bord ou point invalide
            elif bpi[i, j] <= -1:
                classe[i, j] = Terrain.DEPRESSION
            elif bpi[i, j] >= 1:
                classe[i, j] = Terrain.CRETE
            else:
                if -3 < pts[i, j] < 3:
                    classe[i, j] = Terrain.PLAT
                else:
                    classe[i, j] = Terrain.PENTE
    return classe


def classif_2(mnt):
    bpi = calcul_BPI(mnt, r=2)
    classe = np.zeros_like(mnt)
    # parcourir les bpi pour trier les points
    for i in range(bpi.shape[0]):
        for j in range(bpi.shape[1]):
            if np.isnan(bpi[i, j]):
                classe[i, j] = -1  # bord ou point invalide
            elif bpi[i, j] <= -1 or bpi[i, j] >= 1:
                classe[i, j] = Terrain.DUNE
            else:
                classe[i, j] = Terrain.PAS_DUNE
    return classe


