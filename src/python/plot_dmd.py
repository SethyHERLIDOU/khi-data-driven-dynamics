import numpy as np
import matplotlib.pyplot as plt
from pod import build_snapshot_matrix
from dmd import compute_dmd

# --- Configuration ---
data_dir = "../../data/raw"
dt = 0.05  # Le pas de temps d'export de Basilisk
r = 20     # Troncature pour stabiliser la matrice (filtrage du bruit)

# 1. Extraction des données
print("Préparation des données pour la DMD...")
X_fluct, X_mean, _, _ = build_snapshot_matrix(data_dir)

# CRUCIAL : La DMD requiert le champ total, pas seulement les fluctuations
X_raw = X_fluct + X_mean

# 2. Calcul de la DMD
Phi, lambdas, omega, b = compute_dmd(X_raw, dt, r=r)

# 3. Tracé des spectres
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')

# --- Panneau 1 : Le Cercle Unité (Dynamique Discrète) ---
ax1 = axes[0]
theta = np.linspace(0, 2 * np.pi, 200)
ax1.plot(np.cos(theta), np.sin(theta), 'k--', label='Cercle Unité ($|\lambda|=1$)')
ax1.scatter(np.real(lambdas), np.imag(lambdas), c='royalblue', edgecolors='k', s=60, alpha=0.8)

# Décoration du panneau 1
ax1.set_xlabel('Re($\lambda$)')
ax1.set_ylabel('Im($\lambda$)')
ax1.set_title('Spectre Discret (Valeurs propres $\lambda$)')
ax1.axis('equal')
ax1.grid(True, linestyle=':')
ax1.legend()

# --- Panneau 2 : Fréquences vs Taux de croissance (Dynamique Continue) ---
ax2 = axes[1]
frequences = np.imag(omega) / (2 * np.pi)
taux_croissance = np.real(omega)

# L'amplitude physique b détermine la taille des points pour repérer les modes dominants
amplitudes = np.abs(b)
sizes = 300 * (amplitudes / np.max(amplitudes)) + 10 # Normalisation

sc = ax2.scatter(frequences, taux_croissance, s=sizes, c=np.abs(omega), 
                 cmap='plasma', edgecolors='k', alpha=0.8)

# Axes de référence
ax2.axhline(0, color='k', linestyle='--', label='Stabilité neutre ($\omega_r = 0$)')
ax2.axvline(0, color='k', linestyle='-', alpha=0.3)

# Décoration du panneau 2
ax2.set_xlabel('Fréquence $f$')
ax2.set_ylabel('Taux de croissance Re($\omega$)')
ax2.set_title('Spectre Continu DMD')
ax2.grid(True, linestyle=':')
ax2.legend()
plt.colorbar(sc, ax=ax2, label='$|\omega|$')

plt.savefig("dmd_spectrum.png", dpi=300)
print("-> Spectres DMD sauvegardés avec succès (dmd_spectrum.png)")