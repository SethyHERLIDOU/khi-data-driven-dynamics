import numpy as np
import matplotlib.pyplot as plt
from pod import build_snapshot_matrix, compute_pod

# --- Configuration ---
data_dir = "../../data/raw"
L0 = 1.0
r = 4               # Nombre de modes pour la compression
snapshot_idx = -1   # On analyse le dernier instant de la simulation

# 1. Extraction et décomposition
X_fluct, X_mean, grid_shape, files = build_snapshot_matrix(data_dir)
U, lambdas, energie_relative, Vt = compute_pod(X_fluct)

nx, ny = grid_shape
dx = L0 / nx
half_size = nx * ny

# 2. Algorithme de Reconstruction (ROM)
print(f"Reconstruction de l'écoulement avec {r} modes...")
Ur = U[:, :r]

# Calcul des coefficients (a = U^T * X') et reconstruction (X'_recon = U * a)
a_r = np.dot(Ur.T, X_fluct)
X_fluct_recon = np.dot(Ur, a_r)

# Champ total = Fluctuations reconstruites + Moyenne
X_recon = X_fluct_recon + X_mean

# 3. Extraction de l'instant ciblé pour la comparaison
snap_orig = X_fluct[:, snapshot_idx] + X_mean[:, 0] # Vérité terrain
snap_recon = X_recon[:, snapshot_idx]               # Modèle réduit

# Fonction utilitaire pour calculer la vorticité d'un vecteur 1D
def get_vorticity(snap_vector):
    u = snap_vector[:half_size].reshape((ny, nx))
    v = snap_vector[half_size:].reshape((ny, nx))
    dudy = np.gradient(u, dx, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    return dvdx - dudy

vort_orig = get_vorticity(snap_orig)
vort_recon = get_vorticity(snap_recon)
vort_err = np.abs(vort_orig - vort_recon) # Erreur absolue

# 4. Tracé de la preuve de concept
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmax = np.max(np.abs(vort_orig))

# Vérité terrain
im0 = axes[0].imshow(vort_orig, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0].set_title('CFD Originale (Vérité Terrain)')

# Reconstruction
im1 = axes[1].imshow(vort_recon, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1].set_title(f'Reconstruction POD (r={r} modes)')

# Carte d'erreur
im2 = axes[2].imshow(vort_err, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', cmap='Reds', vmin=0, vmax=vmax*0.5)
axes[2].set_title('Erreur Absolue')

for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
plt.colorbar(im0, ax=axes[:2], fraction=0.03, pad=0.04, label='Vorticité')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Erreur')

plt.tight_layout()
plt.savefig("pod_reconstruction.png", dpi=300)
print("-> Preuve de concept sauvegardée (pod_reconstruction.png)")