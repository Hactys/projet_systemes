"""
Fonctions importantes : Analyse de Modèles Numériques de Terrain (MNT)

Ce module regroupe toutes les fonctions clés développées durant la séance 1,
organisées en 5 grandes catégories :
  1. Calcul de gradient (pente brute selon 3 méthodes)
  2. Dérivées de terrain (pente en norme, exposition)
  3. Coefficients d'Evans (pour les courbures)
  4. BPI (Bathymetric/Terrain Position Index)
  5. Courbures et classification de Dikau
  6. Classification BPI + pente (dunes/crêtes/dépressions)
  7. Rugosité de surface

Convention d'axes pour tous les MNT (matrices numpy 2D) :
  - axis 0 (lignes)  → direction y (Nord-Sud)
  - axis 1 (colonnes) → direction x (Est-Ouest)
  - mnt[i, j] = altitude au point (ligne i, colonne j)

Numérotation 3x3 Evans (vue de dessus, z1 en haut à gauche) :
  z1 z2 z3
  z4 z5 z6
  z7 z8 z9
"""

import os
import hashlib

import numpy as np
from scipy.ndimage import convolve, generic_filter


# 1. CALCUL DE GRADIENT

def TPP(mnt, dx=1.0, dy=1.0):
    """
    Gradient par différences finies avant — méthode TPP (Three-Point Plane).

    Utilise uniquement le voisin immédiat à droite (x) et au-dessus (y) du
    pixel central. C'est la méthode la plus simple mais la moins précise :
    elle introduit un biais systématique vers le coin bas-gauche.

    Formules :
        fx = (z_droite  - z_centre) / dx
        fy = (z_dessus  - z_centre) / dy

    Le résultat est tronqué au domaine intérieur commun (bords = NaN).

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude du MNT.
    dx : float
        Résolution horizontale (distance réelle entre deux colonnes adjacentes).
    dy : float
        Résolution verticale (distance réelle entre deux lignes adjacentes).

    Retour
    ------
    matx : np.ndarray (H, W), float
        Composante x du gradient (NaN sur les bords d'1 pixel).
    maty : np.ndarray (H, W), float
        Composante y du gradient (NaN sur les bords d'1 pixel).
    """
    matx = np.full_like(mnt, np.nan, dtype=float)
    maty = np.full_like(mnt, np.nan, dtype=float)

    fx = (mnt[:, 1:] - mnt[:, :-1]) / dx   # (H, W-1)
    fy = (mnt[1:, :] - mnt[:-1, :]) / dy   # (H-1, W)

    # Zone intérieure commune : on perd 1 pixel de chaque côté
    matx[1:-1, 1:-1] = fx[1:-1, :-1]
    maty[1:-1, 1:-1] = fy[:-1, 1:-1]

    return matx, maty


def FCN(mnt, dx=1.0, dy=1.0):
    """
    Gradient par différences finies centrées — méthode FCN (Four-neighbor Central).

    Utilise les deux voisins opposés dans chaque direction, ce qui annule
    l'erreur d'ordre 1 et donne une précision d'ordre 2 en dx²/dy².
    Méthode d'usage courant en géomorphologie numérique.

    Formules :
        fx = (z_{i,j+1} - z_{i,j-1}) / (2·dx)
        fy = (z_{i+1,j} - z_{i-1,j}) / (2·dy)

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude du MNT.
    dx : float
        Résolution horizontale.
    dy : float
        Résolution verticale.

    Retour
    ------
    matx : np.ndarray (H, W), float
        Composante x du gradient (NaN sur les bords d'1 pixel).
    maty : np.ndarray (H, W), float
        Composante y du gradient (NaN sur les bords d'1 pixel).
    """
    matx = np.full_like(mnt, np.nan, dtype=float)
    maty = np.full_like(mnt, np.nan, dtype=float)

    fx = (mnt[:, 2:] - mnt[:, :-2]) / (2 * dx)   # (H, W-2)
    fy = (mnt[2:, :] - mnt[:-2, :]) / (2 * dy)   # (H-2, W)

    matx[1:-1, 1:-1] = fx[1:-1, :]
    maty[1:-1, 1:-1] = fy[:, 1:-1]

    return matx, maty


def Evans(mnt, s=1.0):
    """
    Gradient par la méthode d'Evans (voisinage 3x3, ajustement polynomial).

    Evans (1980) ajuste un polynôme du second degré sur les 9 voisins et
    évalue les dérivées partielles au centre. Cela revient à une pondération
    des différences centrales sur les 3 colonnes/lignes voisines.
    C'est la méthode la plus robuste au bruit parmi les trois implémentées.

    Formules (coefficients D et E du polynôme d'Evans) :
        fx = D = (z3+z6+z9 - z1-z4-z7) / (6·s)
        fy = E = -(z1+z2+z3 - z7-z8-z9) / (6·s)

    Note : le signe de fy est inversé car l'axe des lignes va vers le bas
    (convention image) alors que l'axe y mathématique va vers le haut.

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude du MNT.
    s : float
        Taille d'un pixel (résolution spatiale, en unités cohérentes avec
        les altitudes pour obtenir un gradient sans dimension ou en m/m).

    Retour
    ------
    matx : np.ndarray (H, W), float
        Composante x du gradient (NaN sur les bords d'1 pixel).
    maty : np.ndarray (H, W), float
        Composante y du gradient (NaN sur les bords d'1 pixel).
    """
    matx = np.full_like(mnt, np.nan, dtype=float)
    maty = np.full_like(mnt, np.nan, dtype=float)

    z1 = mnt[:-2, :-2]; z2 = mnt[:-2, 1:-1]; z3 = mnt[:-2, 2:]
    z4 = mnt[1:-1, :-2];                      z6 = mnt[1:-1, 2:]
    z7 = mnt[2:,  :-2]; z8 = mnt[2:,  1:-1]; z9 = mnt[2:,  2:]

    matx[1:-1, 1:-1] = (z3 + z6 + z9 - z1 - z4 - z7) / (6 * s)
    maty[1:-1, 1:-1] = -((z1 + z2 + z3 - z7 - z8 - z9) / (6 * s))

    return matx, maty


# 2. DÉRIVÉES DE TERRAIN

def pente(fx, fy):
    """
    Norme du gradient (pente locale, sans dimension ou en m/m).

    La pente est la magnitude du vecteur gradient. Une valeur de 1.0 correspond
    à une pente à 45°. Pour obtenir les degrés, utiliser np.degrees(np.arctan(pente)).

    Paramètres
    ----------
    fx : np.ndarray, float
        Composante x du gradient (sortie de TPP, FCN ou Evans).
    fy : np.ndarray, float
        Composante y du gradient (sortie de TPP, FCN ou Evans).

    Retour
    ------
    np.ndarray, float
        Matrice de pente ≥ 0, même forme que fx et fy.
    """
    return np.sqrt(fx**2 + fy**2)


def exposition(fx, fy):
    """
    Exposition (aspect) du terrain en radians.

    L'exposition indique l'orientation de la face du terrain par rapport au
    nord (convention géographique). Elle est définie comme l'angle du vecteur
    gradient projeté sur le plan horizontal, compté depuis le nord dans le
    sens horaire.

    Formule : arctan2(-fx, -fy)
    Les signes négatifs tournent de 180° pour pointer vers l'aval (direction
    de la plus grande pente descendante) plutôt que vers l'amont.

    Retourne NaN là où le gradient est nul (terrain plat, pente indéfinie).

    Paramètres
    ----------
    fx : np.ndarray, float
        Composante x du gradient.
    fy : np.ndarray, float
        Composante y du gradient.

    Retour
    ------
    np.ndarray, float
        Angles en radians ∈ [-π, π]. 0 = Nord, π/2 = Est, -π/2 = Ouest.
    """
    return np.arctan2(-fx, -fy)


# 3. COEFFICIENTS D'EVANS (POUR COURBURES)

def coeffs_evans(mnt, s=1.0):
    """
    Calcule les 5 coefficients du polynôme d'Evans sur voisinage 3x3.

    Evans (1980) modélise localement le terrain comme un polynôme du 2nd degré :
        z(x,y) = Ax² + By² + Cxy + Dx + Ey + F

    Les coefficients A à E sont calculés par ajustement sur les 9 voisins.
    Ils sont nécessaires pour calculer les courbures (verticale, horizontale,
    plan, profil) qui décrivent la forme locale du terrain.

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude du MNT.
    s : float
        Taille d'un pixel (résolution spatiale).

    Retour
    ------
    A : np.ndarray (H-2, W-2), float
        Courbure pure en x (dérivée seconde en x). Positif = concave en x.
    B : np.ndarray (H-2, W-2), float
        Courbure pure en y (dérivée seconde en y). Positif = concave en y.
    C : np.ndarray (H-2, W-2), float
        Courbure croisée (torsion). Positif = rotation du gradient vers le NE.
    D : np.ndarray (H-2, W-2), float
        Gradient en x (= fx dans Evans).
    E : np.ndarray (H-2, W-2), float
        Gradient en y (= fy dans Evans).

    Notes
    -----
    Les tableaux retournés ont la taille (H-2, W-2) : ils ne couvrent PAS
    les bords du MNT (1 pixel retiré de chaque côté).
    Pour une utilisation directe dans classification_dikau, les passer à
    courbures_verticale_horizontale.
    """
    z1 = mnt[:-2, :-2]; z2 = mnt[:-2, 1:-1]; z3 = mnt[:-2, 2:]
    z4 = mnt[1:-1, :-2]; z5 = mnt[1:-1, 1:-1]; z6 = mnt[1:-1, 2:]
    z7 = mnt[2:,  :-2]; z8 = mnt[2:,  1:-1]; z9 = mnt[2:,  2:]

    A = (z1 + z3 + z4 + z6 + z7 + z9) / (6*s**2) - (z2 + z5 + z8) / (3*s**2)
    B = (z1 + z2 + z3 + z7 + z8 + z9) / (6*s**2) - (z4 + z5 + z6) / (3*s**2)
    C = (z3 + z7 - z1 - z9) / (4*s**2)
    D = (z3 + z6 + z9 - z1 - z4 - z7) / (6*s**2)
    E = -(z1 + z2 + z3 - z7 - z8 - z9) / (6*s**2)

    return A, B, C, D, E


def courbures_verticale_horizontale(mnt, s=1.0):
    """
    Calcule les courbures verticale (profil) et horizontale (plan) du terrain.

    - Courbure verticale (kv) : courbure dans le plan contenant le vecteur de
      pente et la verticale. Positive = terrain convexe en profil (sommet),
      négative = concave (creux, fond de vallée).
    - Courbure horizontale (kh) : courbure des courbes de niveau. Positive =
      convexe en plan (crête divergente), négative = concave (chenal convergent).

    Ces deux courbures servent de base à la classification de Dikau (9 formes).

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude. Un léger lissage gaussien préalable (sigma≈1)
        est recommandé sur les données bruitées pour éviter les artefacts.
    s : float
        Taille d'un pixel (résolution spatiale).

    Retour
    ------
    kv : np.ndarray (H, W), float
        Courbure verticale. NaN sur les bords (1 pixel) et là où la pente
        est quasi nulle (p < 1e-12) — la courbure verticale est mathématiquement
        indéfinie sur un terrain plat.
    kh : np.ndarray (H, W), float
        Courbure horizontale. Mêmes conventions que kv.
    """
    A, B, C, D, E = coeffs_evans(mnt, s)
    fx, fy = D, E
    fxx, fyy, fxy = 2*A, 2*B, C

    p = fx**2 + fy**2   # norme² du gradient
    q = p + 1

    kv = np.full(mnt.shape, np.nan)
    kh = np.full(mnt.shape, np.nan)

    # On travaille sur la zone intérieure (H-2, W-2)
    masque = p > 1e-12
    kv_inner = np.full_like(fx, np.nan)
    kh_inner = np.full_like(fx, np.nan)

    kv_inner[masque] = -(
        fxx[masque]*fx[masque]**2
        + 2*fxy[masque]*fx[masque]*fy[masque]
        + fyy[masque]*fy[masque]**2
    ) / (p[masque] * np.sqrt(q[masque]**3))

    kh_inner[masque] = -(
        fxx[masque]*fy[masque]**2
        - 2*fxy[masque]*fx[masque]*fy[masque]
        + fyy[masque]*fx[masque]**2
    ) / (p[masque] * np.sqrt(q[masque]))

    kv[1:-1, 1:-1] = kv_inner
    kh[1:-1, 1:-1] = kh_inner

    return kv, kh


# 4. BPI (TERRAIN POSITION INDEX)

def _disk_kernel(r):
    """
    Construit un noyau en forme de disque (anneau excluant le centre).

    Le pixel central (0,0) est exclu : on mesure la différence entre un point
    et ses voisins, pas entre un point et lui-même. Le disque de rayon r permet
    de choisir l'échelle d'analyse : un grand rayon détecte des structures
    larges (dunes), un petit rayon détecte des micro-reliefs.

    Paramètres
    ----------
    r : int
        Rayon du disque en pixels. Le noyau résultant a une taille (2r+1, 2r+1).

    Retour
    ------
    kernel : np.ndarray (2r+1, 2r+1), float
        Valeurs 1.0 pour les pixels dans le disque (0 < d² ≤ r²), 0.0 ailleurs.
    """
    D = 2*r + 1
    di, dj = np.ogrid[-r:r+1, -r:r+1]
    d2 = di**2 + dj**2
    kernel = ((d2 > 0) & (d2 <= r**2)).astype(float)
    return kernel


def moyenne_voisins_disque(mat, r=3):
    """
    Moyenne locale dans un disque de rayon r (NaN-aware).

    Calcule, pour chaque pixel, la moyenne de ses voisins contenus dans un
    disque de rayon r (le pixel central est exclu). Si un pixel du disque est
    NaN (bord ou donnée manquante), le résultat est NaN pour ce pixel : on
    n'extrapole pas sur des voisinages incomplets.

    Paramètres
    ----------
    mat : np.ndarray (H, W), float
        Matrice d'altitude (peut contenir des NaN).
    r : int
        Rayon du disque en pixels.

    Retour
    ------
    np.ndarray (H, W), float
        Matrice des moyennes locales. NaN là où le disque déborde ou contient
        des valeurs manquantes.
    """
    kernel = _disk_kernel(r)
    k = int(kernel.sum())   # nombre de pixels valides dans un disque complet

    mat_filled = np.where(np.isnan(mat), 0.0, mat)
    valid = (~np.isnan(mat)).astype(float)

    sum_vals   = convolve(mat_filled, kernel, mode="constant", cval=0.0)
    count_vals = convolve(valid,      kernel, mode="constant", cval=0.0)

    # N'accepter que les disques entièrement dans le domaine valide
    return np.where(np.round(count_vals) == k, sum_vals / k, np.nan)


def get_stable_hash(arr: np.ndarray) -> str:
    # 1. S'assurer que le tableau est en mémoire continue
    # 2. Utiliser les octets bruts du tableau
    # 3. Inclure la forme (shape) et le type (dtype) pour éviter les collisions

    # On utilise np.ascontiguousarray pour gérer les tableaux non contigus (ex: slices)
    arr_contiguous = np.ascontiguousarray(arr)

    # Création du hash
    hasher = hashlib.sha256()
    hasher.update(arr_contiguous.tobytes())
    hasher.update(str(arr_contiguous.shape).encode())
    hasher.update(str(arr_contiguous.dtype).encode())

    return hasher.hexdigest()


def calcul_BPI(mnt, r=2):
    """
    Indice de Position Topographique (BPI / TPI).

    BPI = altitude du pixel - moyenne de ses voisins dans un disque de rayon r.

    Un BPI positif indique un point surélevé par rapport à son voisinage
    (sommet, crête, haut de dune). Un BPI négatif indique un creux (dépression,
    fond de chenal). Un BPI proche de zéro indique un terrain de transition
    (pente, plateau).

    Le rayon r est le paramètre clé :
      - r faible (2-5) : détection de micro-reliefs.
      - r élevé (20-50) : détection de formes à grande échelle (dunes complètes).

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude du MNT.
    r : int
        Rayon du disque de voisinage en pixels.

    Retour
    ------
    np.ndarray (H, W), float
        Matrice BPI. NaN sur les bords et les zones sans données suffisantes.
    """
    # hachage du mnt pour faire une identification rapide et stable d'une execution à l'autre :
    hash_mnt = get_stable_hash(mnt)

    # les bpi déjà calculés sont stockés dans le dossier `bpi_cache` avec un nom de fichier basé sur le hash du mnt et le rayon r
    cache_dir = "bpi_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"bpi_r{r}_hash{hash_mnt}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)

    bpi = mnt - moyenne_voisins_disque(mnt, r)
    np.save(cache_file, bpi)
    return bpi


# 5. CLASSIFICATION DE DIKAU (COURBURES)

def signe_courbure(val, seuil=0.005):
    """
    Discrétise une courbure continue en trois catégories : concave, plat, convexe.

    Le seuil évite de classifier comme concave/convexe des valeurs proches de
    zéro qui correspondent à un terrain essentiellement plan (artefacts numériques
    ou terrains plats réels). Le choix du seuil est empirique et doit être adapté
    à la résolution et au bruit du MNT.

    Paramètres
    ----------
    val : np.ndarray, float
        Valeurs de courbure (kv ou kh issues de courbures_verticale_horizontale).
    seuil : float
        Valeur absolue en dessous de laquelle la courbure est considérée nulle.
        Valeur typique : 0.001 à 0.01 selon le MNT.

    Retour
    ------
    np.ndarray, int
        -1 = concave, 0 = droit/plan, +1 = convexe.
        Même forme que val.
    """
    res = np.zeros_like(val, dtype=int)
    res[val >  seuil] =  1
    res[val < -seuil] = -1
    return res


def classification_dikau(kv, kh, seuil=0.001):
    """
    Classification morphologique de Dikau (9 formes élémentaires de terrain).

    Combine les signes de la courbure verticale (profil) et horizontale (plan)
    pour attribuer à chaque pixel une des 9 classes de Dikau (1991) :

        kv \\ kh |  convexe (+1) |  droit (0)  | concave (-1)
        ---------|---------------|-------------|-------------
        convexe  |  0 - sommet   | 1 - crête   |  2 - passe
        droit    |  3 - épaulem. | 4 - plan    |  5 - chenal
        concave  |  6 - pied     | 7 - vallée  |  8 - cuvette

    Recommandation : lisser le MNT avec un filtre gaussien (sigma ≈ 1-2 pixels)
    avant de calculer les courbures pour réduire le bruit de mesure.

    Paramètres
    ----------
    kv : np.ndarray (H, W), float
        Courbure verticale (profil), sortie de courbures_verticale_horizontale.
    kh : np.ndarray (H, W), float
        Courbure horizontale (plan), sortie de courbures_verticale_horizontale.
    seuil : float
        Seuil de discrétisation passé à signe_courbure.

    Retour
    ------
    classes : np.ndarray (H, W), int
        Entier 0-8 indiquant la classe de Dikau.
        -1 pour les pixels NaN (bords ou données manquantes).
    """
    sv = signe_courbure(kv, seuil)
    sh = signe_courbure(kh, seuil)

    table = {
        ( 1,  1): 0,   # sommet
        ( 1,  0): 1,   # crête
        ( 1, -1): 2,   # passe
        ( 0,  1): 3,   # épaulement
        ( 0,  0): 4,   # plan
        ( 0, -1): 5,   # chenal
        (-1,  1): 6,   # pied
        (-1,  0): 7,   # vallée
        (-1, -1): 8,   # cuvette
    }

    nan_mask = np.isnan(kv) | np.isnan(kh)
    # Encodage arithmétique : sv ∈ {-1,0,1}, sh ∈ {-1,0,1} → index unique dans [0,8]
    # (sv+1)*3 + (sh+1) donne 0..8 pour les 9 combinaisons
    lut = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])  # concave/plat/convexe × concave/plat/convexe
    idx = (sv + 1) * 3 + (sh + 1)
    classes = lut[idx]
    classes[nan_mask] = -1

    return classes


LABELS_DIKAU = [
    "Sommet", "Crête", "Passe", "Épaulement",
    "Plan", "Chenal", "Pied", "Vallée", "Cuvette"
]


# 6. CLASSIFICATION BPI + PENTE (dunes / crêtes / dépressions)

class Terrain:
    """Constantes entières pour les classes de terrain (classification BPI+pente)."""
    PLAT       = 0
    DEPRESSION = 1
    CRETE      = 2
    PENTE      = 3
    DUNE       = 4
    PAS_DUNE   = 5


def classif_1(mnt):
    """
    Classification large échelle : plat / dépression / crête / pente.

    Utilise le BPI à grand rayon (r=30) pour distinguer les structures majeures,
    puis la pente locale (Evans) pour séparer les zones de transition entre
    plateau et pente. Les seuils (±1 pour BPI, 0.18 pour la pente) ont été
    calibrés empiriquement sur les données de Dunkerque.

    Règle de décision :
        BPI ≤ -1              → DEPRESSION
        BPI ≥ +1              → CRETE
        |BPI| < 1, pente < 0.18 → PLAT
        |BPI| < 1, pente ≥ 0.18 → PENTE

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude (orientation corrigée : ligne 0 = sud).

    Retour
    ------
    classe : np.ndarray (H, W), int
        Classes selon Terrain.* (entiers 0-3). -1 = pixel invalide/bord.
    """
    bpi  = calcul_BPI(mnt, r=30)
    pts  = pente(*Evans(mnt))
    nan_mask = np.isnan(bpi)

    classe = np.full_like(mnt, Terrain.PLAT, dtype=int)
    classe[~nan_mask & (pts >= 0.18)] = Terrain.PENTE
    classe[~nan_mask & (bpi <= -1)]   = Terrain.DEPRESSION
    classe[~nan_mask & (bpi >= 1)]    = Terrain.CRETE
    classe[nan_mask]                  = -1

    return classe


def classif_2(mnt):
    """
    Classification fine échelle : détection de dunes.

    Utilise le BPI à petit rayon (r=7) pour détecter les formes positives ou
    négatives qui correspondent à l'échelle caractéristique des dunes de
    Dunkerque. Un BPI fort (positif ou négatif) signale un relief marqué
    à cette échelle → DUNE. Le reste → PAS_DUNE.

    Paramètres
    ----------
    mnt : np.ndarray (H, W), float
        Matrice d'altitude.

    Retour
    ------
    classe : np.ndarray (H, W), int
        Terrain.DUNE (4) ou Terrain.PAS_DUNE (5). -1 = pixel invalide/bord.
    """
    bpi      = calcul_BPI(mnt, r=7)
    nan_mask = np.isnan(bpi)

    classe = np.full_like(mnt, Terrain.PAS_DUNE, dtype=int)
    classe[~nan_mask & (np.abs(bpi) >= 1)] = Terrain.DUNE
    classe[nan_mask]                        = -1

    return classe


# 7. RUGOSITÉ DE SURFACE

def _nansum_filter(values):
    """
    Somme robuste aux NaN pour scipy.ndimage.generic_filter.

    Retourne NaN si tous les pixels du voisinage sont NaN (zone sans données),
    sinon retourne la somme des valeurs non-NaN. Nécessaire car uniform_filter
    propage les NaN et produit une image entièrement blanche dans ce cas.

    Paramètres
    ----------
    values : np.ndarray 1D, float
        Pixels du voisinage aplati par generic_filter.

    Retour
    ------
    float
        Somme ou NaN.
    """
    return np.nan if np.all(np.isnan(values)) else float(np.nansum(values))


def rugosite(mat_pente_rad, mat_expo_rad, taille_voisinage=3):
    """
    Indice de rugosité par dispersion des vecteurs normaux (méthode vectorielle).

    Pour chaque pixel, on convertit la pente et l'exposition en un vecteur
    normal unitaire à la surface, puis on somme ces vecteurs sur un voisinage
    carré. Un terrain lisse donne des vecteurs alignés → norme de la somme ≈ n.
    Un terrain rugueux donne des vecteurs dispersés → norme faible.

    L'indice k = 1 - ||R|| / n ∈ [0, 1] :
        k ≈ 0 : terrain plat/lisse
        k ≈ 1 : terrain très rugueux (normales très dispersées)

    Composantes du vecteur normal unitaire :
        x = sin(pente) · cos(exposition)
        y = sin(pente) · sin(exposition)
        z = cos(pente)

    Paramètres
    ----------
    mat_pente_rad : np.ndarray (H, W), float
        Pente en radians (= arctan(||gradient||)).
    mat_expo_rad : np.ndarray (H, W), float
        Exposition en radians (sortie de la fonction exposition).
    taille_voisinage : int (impair recommandé)
        Taille du carré de voisinage (ex : 3 → 3x3 = 9 pixels). Plus la
        fenêtre est grande, plus on mesure la rugosité à grande échelle.

    Retour
    ------
    k : np.ndarray (H, W), float
        Indice de rugosité ∈ [0, 1]. NaN sur les bords et là où pente ou
        exposition est NaN.
    """
    n = taille_voisinage**2

    nx = np.sin(mat_pente_rad) * np.cos(mat_expo_rad)
    ny = np.sin(mat_pente_rad) * np.sin(mat_expo_rad)
    nz = np.cos(mat_pente_rad)

    kw = dict(size=taille_voisinage, mode="reflect")
    nx_bar = generic_filter(nx, _nansum_filter, **kw)
    ny_bar = generic_filter(ny, _nansum_filter, **kw)
    nz_bar = generic_filter(nz, _nansum_filter, **kw)

    r = np.sqrt(nx_bar**2 + ny_bar**2 + nz_bar**2)
    k = np.clip(1 - r / n, 0, 1)

    # Masque NaN
    nan_mask = np.isnan(mat_pente_rad) | np.isnan(mat_expo_rad)
    k[nan_mask] = np.nan

    # Bords du filtre → NaN (voisinage incomplet)
    d = taille_voisinage // 2
    if d > 0:
        k[:d,  :] = np.nan
        k[-d:, :] = np.nan
        k[:,  :d] = np.nan
        k[:, -d:] = np.nan

    return k


def rugosite_std_locale(mat, taille=3):
    """
    Rugosité par écart-type local (méthode statistique simple).

    Calcule l'écart-type des valeurs d'altitude (ou de pente) dans un voisinage
    carré autour de chaque pixel. Contrairement à la méthode vectorielle, elle
    ne tient pas compte de l'orientation — elle mesure la variabilité brute des
    hauteurs locales.

    Utilisation typique : passer la matrice de pente (sans gradient préalable)
    pour mesurer la variabilité de la pente plutôt que de l'altitude.

    Paramètres
    ----------
    mat : np.ndarray (H, W), float
        Matrice d'entrée (altitude, pente, ou autre dérivée du MNT).
    taille : int
        Taille du carré de voisinage (ex : 3 → 3x3, 15 → 15x15).

    Retour
    ------
    np.ndarray (H, W), float
        Écart-type local. Valeur élevée = relief variable = rugueux.
    """
    return generic_filter(mat, np.std, size=taille, mode="constant", cval=np.nan)


def extraire_fenetre_centree(coord, matrice, taille):
    """
    Extrait une fenêtre carrée centrée sur un pixel donné.

    Retourne NaN si la fenêtre déborde des bords de la matrice (mode strict :
    pas d'extrapolation, pas de rembourrage).

    Paramètres
    ----------
    coord : tuple (int, int)
        Coordonnées (ligne, colonne) du pixel central.
    matrice : np.ndarray (H, W), float
        Matrice source.
    taille : int (impair)
        Côté de la fenêtre carrée.

    Retour
    ------
    np.ndarray (taille, taille), float
        Fenêtre extraite, ou tableau rempli de NaN si le pixel est trop proche
        du bord pour qu'une fenêtre complète soit disponible.
    """
    l_centre, c_centre = coord
    d = taille // 2
    H, W = matrice.shape

    l_min, l_max = l_centre - d, l_centre + d + 1
    c_min, c_max = c_centre - d, c_centre + d + 1

    if l_min < 0 or l_max > H or c_min < 0 or c_max > W:
        return np.full((taille, taille), np.nan)

    return matrice[l_min:l_max, c_min:c_max]
