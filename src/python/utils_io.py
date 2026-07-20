import numpy as np

def read_snapshot(filepath):
    """
    Lit un fichier binaire issu de Basilisk et retourne les champs u et v.
    """
    with open(filepath, 'rb') as f:
        # Lecture de l'en-tête (N_export, N_export)
        header = np.fromfile(f, dtype=np.int32, count=2)
        nx, ny = header[0], header[1]
        
        # Lecture du corps des données (intercalé ux, uy, ux, uy...)
        data = np.fromfile(f, dtype=np.float64)
        
    # Restructuration en matrices 2D
    # Basilisk a exporté ligne par ligne (y puis x)
    u = data[0::2].reshape((ny, nx))
    v = data[1::2].reshape((ny, nx))
    
    return u, v