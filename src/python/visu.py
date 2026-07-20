import numpy as np
import matplotlib.pyplot as plt
from utils_io import read_snapshot

# Paramètres du domaine physique
L0 = 1.0
# On cible un des derniers fichiers générés (vérifie que ce fichier existe dans data/raw/)
snap_path = "../../data/raw/snap_04.00.bin"

print(f"Lecture du fichier : {snap_path}")
u, v = read_snapshot(snap_path)
nx, ny = u.shape
dx = L0 / nx

# Calcul de la vorticité par différences finies centrales
# np.gradient(tableau, pas, axe) : l'axe 0 est Y (lignes), l'axe 1 est X (colonnes)
dudy = np.gradient(u, dx, axis=0)
dvdx = np.gradient(v, dx, axis=1)
vorticity = dvdx - dudy

# Tracé
plt.figure(figsize=(8, 6))
plt.imshow(vorticity, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', cmap='RdBu_r')
plt.colorbar(label='Vorticité $\omega$')
plt.title('Instabilité de Kelvin-Helmholtz (t=4.0)')
plt.xlabel('x')
plt.ylabel('y')

# Sauvegarde
out_img = "khi_vorticity.png"
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Superbe ! L'image a été sauvegardée sous : {out_img}")