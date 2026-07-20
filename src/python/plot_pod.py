import numpy as np
import matplotlib.pyplot as plt
from pod import build_snapshot_matrix, compute_pod

# Configuration
data_dir = "../../data/raw"
L0 = 1.0

# 1. Extraction et calcul POD
X_fluct, X_mean, grid_shape, files = build_snapshot_matrix(data_dir)
U, lambdas, energie_relative, Vt = compute_pod(X_fluct)

nx, ny = grid_shape
dx = L0 / nx

# 2. Tracé du spectre d'énergie (Critère de Pareto)
energie_cumulee = np.cumsum(energie_relative)
n_modes_to_plot = 15

plt.figure(figsize=(10, 4))

# Énergie individuelle
plt.subplot(1, 2, 1)
plt.plot(range(1, n_modes_to_plot + 1), energie_relative[:n_modes_to_plot], 'bo-', label='Énergie par mode')
plt.ylabel('Énergie relative')
plt.xlabel('Numéro du mode (i)')
plt.grid(True)

# Énergie cumulée
plt.subplot(1, 2, 2)
plt.plot(range(1, n_modes_to_plot + 1), energie_cumulee[:n_modes_to_plot], 'ro-', label='Énergie cumulée')
plt.axhline(0.9, color='k', linestyle='--', label='90% de l\'énergie')
plt.ylabel('Énergie cumulée')
plt.xlabel('Numéro du mode (i)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pod_spectrum.png", dpi=300)
print("-> Spectre d'énergie sauvegardé (pod_spectrum.png)")

# 3. Tracé des premiers modes spatiaux (Vorticité)
# On trace les 3 premiers modes
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
half_size = nx * ny

for i in range(3):
    mode_vector = U[:, i]
    
    # Séparation u/v et passage en 2D
    u_mode = mode_vector[:half_size].reshape((ny, nx))
    v_mode = mode_vector[half_size:].reshape((ny, nx))
    
    # Calcul de la vorticité du mode spatial
    dudy = np.gradient(u_mode, dx, axis=0)
    dvdx = np.gradient(v_mode, dx, axis=1)
    vorticity_mode = dvdx - dudy
    
    ax = axes[i]
    # On centre la colormap sur 0 avec vmin/vmax symétriques
    vmax = np.max(np.abs(vorticity_mode))
    im = ax.imshow(vorticity_mode, extent=[-0.5, 0.5, -0.5, 0.5], 
                   origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Mode {i+1} ({energie_relative[i]*100:.1f}%)')
    ax.set_xlabel('x')
    if i == 0:
        ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("pod_spatial_modes.png", dpi=300)
print("-> Modes spatiaux sauvegardés (pod_spatial_modes.png)")