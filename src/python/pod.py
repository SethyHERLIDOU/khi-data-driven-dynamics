import numpy as np
import glob
import os
from utils_io import read_snapshot

def build_snapshot_matrix(data_dir):
    """Construit la matrice des fluctuations X' à partir des fichiers .bin"""
    # Recherche et tri alphabétique des fichiers (ordre chronologique)
    files = sorted(glob.glob(os.path.join(data_dir, "snap_*.bin")))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .bin trouvé dans {data_dir}")
    
    snapshots = []
    print(f"Chargement de {len(files)} snapshots...")
    
    for f in files:
        u, v = read_snapshot(f)
        # Aplatissement et concaténation : un vecteur 1D par instant t
        snap_vector = np.concatenate((u.ravel(), v.ravel()))
        snapshots.append(snap_vector)
        
    # Transposition : chaque colonne est un temps t, chaque ligne un point spatial
    X = np.column_stack(snapshots)
    
    # Soustraction de la moyenne (Reynolds decomposition)
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_fluct = X - X_mean
    
    # On sauvegarde les dimensions d'origine pour pouvoir reformer les images plus tard
    nx, ny = u.shape
    grid_shape = (nx, ny)
    
    return X_fluct, X_mean, grid_shape, files

def compute_pod(X_fluct):
    """Calcule la Proper Orthogonal Decomposition par SVD économique."""
    print(f"Calcul POD sur matrice {X_fluct.shape} (Ns x Nt)...")
    
    # SVD économique
    U, S, Vt = np.linalg.svd(X_fluct, full_matrices=False)
    
    # Calcul de l'énergie (valeurs propres de la matrice de covariance)
    lambdas = (S ** 2) / (X_fluct.shape[1] - 1)
    energie_totale = np.sum(lambdas)
    energie_relative = lambdas / energie_totale
    
    print("SVD terminée avec succès.")
    return U, lambdas, energie_relative, Vt