import numpy as np

def compute_dmd(X, dt, r=None):
    """
    Calcule la Dynamic Mode Decomposition (Exact DMD).
    """
    print(f"Lancement de l'algorithme DMD sur {X.shape} (Ns x Nt)...")

    # 1. Création des matrices décalées temporellement
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # 2. SVD tronquée sur X1 pour réduire la dimensionnalité
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    
    # Troncature au rang r (pour filtrer le bruit numérique de Basilisk)
    if r is not None:
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]

    # 3. Calcul de l'opérateur linéaire réduit (Atilde)
    S_inv = np.diag(1.0 / S)
    Atilde = np.linalg.multi_dot([U.T, X2, Vt.T, S_inv])

    # 4. Décomposition aux valeurs propres de Atilde
    eigenvalues, W = np.linalg.eig(Atilde)

    # 5. Calcul des Modes DMD Exacts (Phi)
    # Formule : Phi = X2 * V * Sigma^-1 * W
    Phi = np.linalg.multi_dot([X2, Vt.T, S_inv, W])

    # 6. Dynamique continue (transformation du plan discret au plan continu)
    omega = np.log(eigenvalues) / dt

    # 7. Calcul des amplitudes initiales b (projection du premier snapshot)
    # X_0 = Phi * b  =>  b = pseudo_inverse(Phi) * X_0
    b = np.linalg.pinv(Phi).dot(X1[:, 0])

    print("DMD terminée avec succès.")
    return Phi, eigenvalues, omega, b