#include "navier-stokes/centered.h"

// --- Paramètres physiques ---
double U0 = 1.0;            // Vitesse caractéristique
double delta_shear = 0.05;  // Épaisseur de la couche de cisaillement
double Re = 10000.;         // Nombre de Reynolds

// --- Paramètres numériques ---
int maxlevel = 9;           // Niveau de raffinement max (équivalent grille 512x512)
double tend = 4.0;          // Temps de fin de simulation

int main() {
    L0 = 1.0;               // Taille du domaine temporel [0, 1]
    origin (-0.5, -0.5);    // Centrage du domaine sur (0,0)
    
    // Conditions aux limites
    periodic (right);       // Périodique en X (les tourbillons bouclent)
    
    // Parois glissantes (slip) en haut et en bas
    u.n[bottom] = dirichlet(0.);
    u.t[bottom] = neumann(0.);
    u.n[top]    = dirichlet(0.);
    u.t[top]    = neumann(0.);

    // Viscosité cinématique nu = U0 * L0 / Re
    const face vector muc[] = {1./Re, 1./Re};
    mu = muc;

    init_grid (1 << 7);     // Grille de départ 128x128
    run();
}

// --- Initialisation à t = 0 ---
event init (t = 0) {
    // Mode m=2 (deux vagues dans le domaine)
    double k = 2. * M_PI * 2.; 
    double amp = 0.05 * U0; // Amplitude de la perturbation (5%)

    foreach() {
        // u(y) = U0 * tanh(y/delta)
        u.x[] = U0 * tanh(y / delta_shear);
        // v(x,y) = amp * sin(kx) * exp(-y^2/delta^2)
        u.y[] = amp * sin(k * x) * exp(-(y*y)/(delta_shear*delta_shear));
    }
    boundary ((scalar *){u});
}

// --- Raffinement de maillage adaptatif (AMR) ---
event adapt (i++) {
    // On demande à Basilisk de raffiner là où le gradient de vitesse est fort
    adapt_wavelet ((scalar *){u}, (double[]){1e-3, 1e-3}, maxlevel, 5);
}

// --- L'interface avec Python : Export Binaire ---
event export_data (t += 0.05; t <= tend) {
    char name[100];
    // On remonte d'un cran (..) pour aller dans data/raw/
    sprintf (name, "../../data/raw/snap_%05.2f.bin", t);
    FILE * fp = fopen (name, "wb");

    int N_export = 256; // Résolution d'export pour Python (matrice 256x256)
    double step = L0 / N_export;

    // En-tête : on écrit les dimensions pour aider la lecture Python
    fwrite(&N_export, sizeof(int), 1, fp);
    fwrite(&N_export, sizeof(int), 1, fp);

    // Interpolation de la grille adaptative vers une grille cartésienne stricte
    for (int j = 0; j < N_export; j++) {
        double yp = Y0 + j * step + step / 2.;
        for (int i = 0; i < N_export; i++) {
            double xp = X0 + i * step + step / 2.;
            
            double ux = interpolate (u.x, xp, yp);
            double uy = interpolate (u.y, xp, yp);
            
            // Écriture des données
            fwrite(&ux, sizeof(double), 1, fp);
            fwrite(&uy, sizeof(double), 1, fp);
        }
    }
    fclose(fp);
    
    // Petit affichage dans le terminal pour suivre l'avancement
    printf("Export t = %.2f terminé.\n", t);
}