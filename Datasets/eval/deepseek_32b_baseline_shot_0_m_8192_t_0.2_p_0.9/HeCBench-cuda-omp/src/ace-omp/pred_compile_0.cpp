#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400

typedef double nRarray[DATAYSIZE][DATAXSIZE];

#define SQ(x) ((x)*(x))

#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

double dFphi(double phi, double u, double lambda) {
    return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

double GradientX(double phi[DATAZSIZE][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z) {
    return (phi[z][y][x+1] - phi[z][y][x-1]) / (2.0*dx);
}

double GradientY(double phi[DATAZSIZE][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z) {
    return (phi[z][y+1][x] - phi[z][y-1][x]) / (2.0*dy);
}

double GradientZ(double phi[DATAZSIZE][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z) {
    return (phi[z+1][y][x] - phi[z-1][y][x]) / (2.0*dz);
}

double Divergence(double phix[DATAZSIZE][DATAYSIZE][DATAXSIZE], 
                  double phiy[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                  double phiz[DATAZSIZE][DATAYSIZE][DATAXSIZE], 
                  double dx, double dy, double dz, int x, int y, int z) {
    return GradientX(phix, dx, dy, dz, x, y, z) +
           GradientY(phiy, dx, dy, dz, x, y, z) +
           GradientZ(phiz, dx, dy, dz, x, y, z);
}

double Laplacian(double phi[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z) {
    double phixx = (phi[z][y][x+1] + phi[z][y][x-1] - 2.0 * phi[z][y][x]) / SQ(dx);
    double phiyy = (phi[z][y+1][x] + phi[z][y-1][x] - 2.0 * phi[z][y][x]) / SQ(dy);
    double phizz = (phi[z+1][y][x] + phi[z-1][y][x] - 2.0 * phi[z][y][x]) / SQ(dz);
    return phixx + phiyy + phizz;
}

double An(double phix, double phiy, double phiz, double epsilon) {
    if (phix != 0.0 || phiy != 0.0 || phiz != 0.0) {
        return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon)) *
               ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
               ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
    } else {
        return (1.0 - ((5.0/3.0)*epsilon));
    }
}

double Wn(double phix, double phiy, double phiz, double epsilon, double W0) {
    return (W0 * An(phix, phiy, phiz, epsilon));
}

double taun(double phix, double phiy, double phiz, double epsilon, double tau0) {
    return tau0 * SQ(An(phix, phiy, phiz, epsilon));
}

double dFunc(double l, double m, double n) {
    if (l != 0.0 || m != 0.0 || n != 0.0) {
        return (((l*l*l*(SQ(m)+SQ(n))) - (l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
                ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
    } else {
        return 0.0;
    }
}

void calculateForce(double phi[DATAZSIZE][DATAYSIZE][DATAXSIZE], 
                    double Fx[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                    double Fy[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                    double Fz[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0) {
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < DATAZSIZE; z++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int x = 0; x < DATAXSIZE; x++) {
                if ((x < (DATAXSIZE-1)) && (y < (DATAYSIZE-1)) &&
                    (z < (DATAZSIZE-1)) && (x > 0) && (y > 0) && (z > 0)) {
                    double phix = GradientX(phi, dx, dy, dz, x, y, z);
                    double phiy = GradientY(phi, dx, dy, dz, x, y, z);
                    double phiz = GradientZ(phi, dx, dy, dz, x, y, z);
                    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
                    double c = 16.0 * W0 * epsilon;
                    double w = Wn(phix, phiy, phiz, epsilon, W0);
                    double w2 = SQ(w);
                    
                    Fx[z][y][x] = w2 * phix + sqGphi * w * c * dFunc(phix, phiy, phiz);
                    Fy[z][y][x] = w2 * phiy + sqGphi * w * c * dFunc(phiy, phiz, phix);
                    Fz[z][y][x] = w2 * phiz + sqGphi * w * c * dFunc(phiz, phix, phiy);
                } else {
                    Fx[z][y][x] = 0.0;
                    Fy[z][y][x] = 0.0;
                    Fz[z][y][x] = 0.0;
                }
            }
        }
    }
}

void allenCahn(double phinew[DATAZSIZE][DATAYSIZE][DATAXSIZE], 
               double phiold[DATAZSIZE][DATAYSIZE][DATAXSIZE],
               double uold[DATAZSIZE][DATAYSIZE][DATAXSIZE],
               double Fx[DATAZSIZE][DATAYSIZE][DATAXSIZE],
               double Fy[DATAZSIZE][DATAYSIZE][DATAXSIZE],
               double Fz[DATAZSIZE][DATAYSIZE][DATAXSIZE],
               double epsilon, double W0, double tau0, double lambda,
               double dt, double dx, double dy, double dz) {
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < DATAZSIZE; z++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int x = 0; x < DATAXSIZE; x++) {
                if ((x < (DATAXSIZE-1)) && (y < (DATAYSIZE-1)) &&
                    (z < (DATAZSIZE-1)) && (x > 0) && (y > 0) && (z > 0)) {
                    double phix = GradientX(phiold, dx, dy, dz, x, y, z);
                    double phiy = GradientY(phiold, dx, dy, dz, x, y, z);
                    double phiz = GradientZ(phiold, dx, dy, dz, x, y, z);
                    
                    phinew[z][y][x] = phiold[z][y][x] + 
                                     (dt / taun(phix, phiy, phiz, epsilon, tau0)) *
                                     (Divergence(Fx, Fy, Fz, dx, dy, dz, x, y, z) -
                                      dFphi(phiold[z][y][x], uold[z][y][x], lambda));
                }
            }
        }
    }
}

void boundaryConditionsPhi(double phinew[DATAZSIZE][DATAYSIZE][DATAXSIZE]) {
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < DATAZSIZE; z++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int x = 0; x < DATAXSIZE; x++) {
                if (x == 0 || x == DATAXSIZE-1 || y == 0 || y == DATAYSIZE-1 ||
                    z == 0 || z == DATAZSIZE-1) {
                    phinew[z][y][x] = -1.0;
                }
            }
        }
    }
}

void thermalEquation(double unew[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                    double uold[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                    double phinew[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                    double phiold[DATAZSIZE][DATAYSIZE][DATAXSIZE],
                    double D, double dt, double dx, double dy, double dz) {
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < DATAZSIZE; z++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int x = 0; x < DATAXSIZE; x++) {
                if ((x < (DATAXSIZE-1)) && (y < (DATAYSIZE-1)) &&
                    (z < (DATAZSIZE-1)) && (x > 0) && (y > 0) && (z > 0)) {
                    unew[z][y][x] = uold[z][y][x] + 
                                    0.5 * (phinew[z][y][x] - phiold[z][y][x]) +
                                    dt * D * Laplacian(uold, dx, dy, dz, x, y, z);
                }
            }
        }
    }
}

void boundaryConditionsU(double unew[DATAZSIZE][DATAYSIZE][DATAXSIZE], double delta) {
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < DATAZSIZE; z++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int x = 0; x < DATAXSIZE; x++) {
                if (x == 0 || x == DATAXSIZE-1 || y == 0 || y == DATAYSIZE-1 ||
                    z == 0 || z == DATAZSIZE-1) {
                    unew[z][y][x] = -delta;
                }
            }
        }
    }
}

void swapGrid(double cnew[DATAZSIZE][DATAYSIZE][DATAXSIZE],
              double cold[DATAZSIZE][DATAYSIZE][DATAXSIZE]) {
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < DATAZSIZE; z++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int x = 0; x < DATAXSIZE; x++) {
                double tmp = cnew[z][y][x];
                cnew[z][y][x] = cold[z][y][x];
                cold[z][y][x] = tmp;
            }
        }
    }
}

void initializationPhi(double phi[DATAZSIZE][DATAYSIZE][DATAXSIZE], double r0) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < DATAXSIZE; x++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int z = 0; z < DATAZSIZE; z++) {
                double dx = x - 0.5 * DATAXSIZE;
                double dy = y - 0.5 * DATAYSIZE;
                double dz = z - 0.5 * DATAZSIZE;
                double r = sqrt(SQ(dx) + SQ(dy) + SQ(dz));
                if (r < r0) {
                    phi[z][y][x] = 1.0;
                } else {
                    phi[z][y][x] = -1.0;
                }
            }
        }
    }
}

void initializationU(double u[DATAZSIZE][DATAYSIZE][DATAXSIZE], double r0, double delta) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < DATAXSIZE; x++) {
        for (int y = 0; y < DATAYSIZE; y++) {
            for (int z = 0; z < DATAZSIZE; z++) {
                double dx = x - 0.5 * DATAXSIZE;
                double dy = y - 0.5 * DATAYSIZE;
                double dz = z - 0.5 * DATAZSIZE;
                double r = sqrt(SQ(dx) + SQ(dy) + SQ(dz));
                if (r < r0) {
                    u[z][y][x] = 0.0;
                } else {
                    u[z][y][x] = -delta * (1.0 - exp(-(r - r0)));
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const int num_steps = atoi(argv[1]);

    const double dx = 0.4;
    const double dy = 0.4;
    const double dz = 0.4;
    const double dt = 0.01;
    const double delta = 0.8;
    const double r0 = 5.0;
    const double epsilon = 0.07;
    const double W0 = 1.0;
    const double beta0 = 0.0;
    const double D = 2.0;
    const double d0 = 0.5;
    const double a1 = 1.25 / sqrt(2.0);
    const double a2 = 0.64;
    const double lambda = (W0 * a1) / d0;
    const double tau0 = ((W0 * W0 * W0 * a1 * a2) / (d0 * D)) + ((W0 * W0 * beta0) / d0);

    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;
    const int vol = nx * ny * nz;
    const size_t vol_in_bytes = sizeof(double) * vol;

    nRarray *phi_host = (nRarray *)malloc(vol_in_bytes);
    nRarray *u_host = (nRarray *)malloc(vol_in_bytes);

    initializationPhi(phi_host, r0);
    initializationU(u_host, r0, delta);

#ifdef VERIFY
    nRarray *phi_ref = (nRarray *)malloc(vol_in_bytes);
    nRarray *u_ref = (nRarray *)malloc(vol_in_bytes);
    memcpy(phi_ref, phi_host, vol_in_bytes);
    memcpy(u_ref, u_host, vol_in_bytes);
    reference(phi_ref, u_ref, vol, num_steps);
#endif

    auto offload_start = std::chrono::steady_clock::now();

    double *d_phiold = phi_host;
    double *d_phinew = (double *)malloc(vol_in_bytes);
    double *d_uold = u_host;
    double *d_unew = (double *)malloc(vol_in_bytes);
    double *d_Fx = (double *)malloc(vol_in_bytes);
    double *d_Fy = (double *)malloc(vol_in_bytes);
    double *d_Fz = (double *)malloc(vol_in_bytes);

    int t = 0;

    auto start = std::chrono::steady_clock::now();

    while (t <= num_steps) {
        calculateForce((double (*)[DATAYSIZE][DATAXSIZE])d_phiold,
                      (double (*)[DATAYSIZE][DATAXSIZE])d_Fx,
                      (double (*)[DATAYSIZE][DATAXSIZE])d_Fy,
                      (double (*)[DATAYSIZE][DATAXSIZE])d_Fz,
                      dx, dy, dz, epsilon, W0, tau0);

        allenCahn((double (*)[DATAYSIZE][DATAXSIZE])d_phinew,
                  (double (*)[DATAYSIZE][DATAXSIZE])d_phiold,
                  (double (*)[DATAYSIZE][DATAXSIZE])d_uold,
                  (double (*)[DATAYSIZE][DATAXSIZE])d_Fx,
                  (double (*)[DATAYSIZE][DATAXSIZE])d_Fy,
                  (double (*)[DATAYSIZE][DATAXSIZE])d_Fz,
                  epsilon, W0, tau0, lambda, dt, dx, dy, dz);

        boundaryConditionsPhi((double (*)[DATAYSIZE][DATAXSIZE])d_phinew);

        thermalEquation((double (*)[DATAYSIZE][DATAXSIZE])d_unew,
                       (double (*)[DATAYSIZE][DATAXSIZE])d_uold,
                       (double (*)[DATAYSIZE][DATAXSIZE])d_phinew,
                       (double (*)[DATAYSIZE][DATAXSIZE])d_phiold,
                       D, dt, dx, dy, dz);

        boundaryConditionsU((double (*)[DATAYSIZE][DATAXSIZE])d_unew, delta);

        swapGrid((double (*)[DATAYSIZE][DATAXSIZE])d_phinew,
                 (double (*)[DATAYSIZE][DATAXSIZE])d_phiold);

        swapGrid((double (*)[DATAYSIZE][DATAXSIZE])d_unew,
                 (double (*)[DATAYSIZE][DATAXSIZE])d_uold);

        t++;
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);

    memcpy(phi_host, d_phiold, vol_in_bytes);
    memcpy(u_host, d_uold, vol_in_bytes);

    free(d_Fx);
    free(d_Fy);
    free(d_Fz);
    free(d_phinew);
    free(d_unew);

    auto offload_end = std::chrono::steady_clock::now();
    auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
    printf("Offload time: %.3f (ms)\n", offload_time * 1e-6f);

#ifdef VERIFY
    bool ok = true;
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                if (fabs(phi_ref[z][y][x] - phi_host[z][y][x]) > 1e-3) {
                    ok = false;
                    printf("phi: %lf %lf\n", phi_ref[z][y][x], phi_host[z][y][x]);
                }
                if (fabs(u_ref[z][y][x] - u_host[z][y][x]) > 1e-3) {
                    ok = false;
                    printf("u: %lf %lf\n", u_ref[z][y][x], u_host[z][y][x]);
                }
            }
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(phi_ref);
    free(u_ref);
#endif

    free(phi_host);
    free(u_host);
    return 0;
}