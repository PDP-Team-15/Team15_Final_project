#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400
#define SQ(x) ((x)*(x))

typedef double nRarray[DATAYSIZE][DATAXSIZE];

#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

__device__ double dFphi(double phi, double u, double lambda) {
    return (-phi * (1.0 - phi * phi) + lambda * u * (1.0 - phi * phi) * (1.0 - phi * phi));
}

__device__ double GradientX(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    return (phi[idx + DATAXSIZE] - phi[idx - DATAXSIZE]) / (2.0 * dx);
}

__device__ double GradientY(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    return (phi[idx + DATAYSIZE * DATAXSIZE] - phi[idx - DATAYSIZE * DATAXSIZE]) / (2.0 * dy);
}

__device__ double GradientZ(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    return (phi[idx + DATAYSIZE * DATAXSIZE * DATAZSIZE] - phi[idx - DATAYSIZE * DATAXSIZE * DATAZSIZE]) / (2.0 * dz);
}

__device__ double Divergence(double *phix, double *phiy, double *phiz, double dx, double dy, double dz, int x, int y, int z) {
    return GradientX(phix, dx, dy, dz, x, y, z) + GradientY(phiy, dx, dy, dz, x, y, z) + GradientZ(phiz, dx, dy, dz, x, y, z);
}

__device__ double Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    double phixx = (phi[idx + DATAXSIZE] + phi[idx - DATAXSIZE] - 2.0 * phi[idx]) / SQ(dx);
    double phiyy = (phi[idx + DATAYSIZE * DATAXSIZE] + phi[idx - DATAYSIZE * DATAXSIZE] - 2.0 * phi[idx]) / SQ(dy);
    double phizz = (phi[idx + DATAYSIZE * DATAXSIZE * DATAZSIZE] + phi[idx - DATAYSIZE * DATAXSIZE * DATAZSIZE] - 2.0 * phi[idx]) / SQ(dz);
    return phixx + phiyy + phizz;
}

__device__ double An(double phix, double phiy, double phiz, double epsilon) {
    if (phix != 0.0 || phiy != 0.0 || phiz != 0.0) {
        double numerator = 1.0 - 3.0 * epsilon;
        double denominator = 1.0 - 3.0 * epsilon;
        double term = (4.0 * epsilon) / denominator;
        double a = SQ(phix) * SQ(phix) + SQ(phiy) * SQ(phiy) + SQ(phiz) * SQ(phiz);
        double b = SQ(phix) + SQ(phiy) + SQ(phiz);
        double c = a / (b * b);
        return numerator * (1.0 + term * c);
    } else {
        return 1.0 - (5.0 / 3.0) * epsilon;
    }
}

__device__ double Wn(double phix, double phiy, double phiz, double epsilon, double W0) {
    return W0 * An(phix, phiy, phiz, epsilon);
}

__device__ double taun(double phix, double phiy, double phiz, double epsilon, double tau0) {
    return tau0 * SQ(An(phix, phiy, phiz, epsilon));
}

__device__ double dFunc(double l, double m, double n) {
    if (l != 0.0 || m != 0.0 || n != 0.0) {
        double numerator = (l * l * l * (SQ(m) + SQ(n)) - l * (SQ(m) * SQ(m) + SQ(n) * SQ(n)));
        double denominator = (SQ(l) + SQ(m) + SQ(n)) * (SQ(l) + SQ(m) + SQ(n));
        return numerator / denominator;
    } else {
        return 0.0;
    }
}

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, double epsilon, double W0, double tau0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x > 0 && x < DATAXSIZE - 1 && y > 0 && y < DATAYSIZE - 1 && z > 0 && z < DATAZSIZE - 1) {
            double phix = GradientX(phi, dx, dy, dz, x, y, z);
            double phiy = GradientY(phi, dx, dy, dz, x, y, z);
            double phiz = GradientZ(phi, dx, dy, dz, x, y, z);
            double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
            double c = 16.0 * W0 * epsilon;
            double w = Wn(phix, phiy, phiz, epsilon, W0);
            double w2 = SQ(w);

            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            Fx[idx] = w2 * phix + sqGphi * w * c * dFunc(phix, phiy, phiz);
            Fy[idx] = w2 * phiy + sqGphi * w * c * dFunc(phiy, phiz, phix);
            Fz[idx] = w2 * phiz + sqGphi * w * c * dFunc(phiz, phix, phiy);
        } else {
            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            Fx[idx] = 0.0;
            Fy[idx] = 0.0;
            Fz[idx] = 0.0;
        }
    }
}

__global__ void allenCahn(double *phinew, double *phiold, double *uold, double *Fx, double *Fy, double *Fz, double epsilon, double W0, double tau0, double lambda, double dt, double dx, double dy, double dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x > 0 && x < DATAXSIZE - 1 && y > 0 && y < DATAYSIZE - 1 && z > 0 && z < DATAZSIZE - 1) {
            double phix = GradientX(phiold, dx, dy, dz, x, y, z);
            double phiy = GradientY(phiold, dx, dy, dz, x, y, z);
            double phiz = GradientZ(phiold, dx, dy, dz, x, y, z);
            double divergence = Divergence(Fx, Fy, Fz, dx, dy, dz, x, y, z);
            double dfphi = dFphi(phiold[x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z], uold[x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z], lambda);
            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            phinew[idx] = phiold[idx] + (dt / taun(phix, phiy, phiz, epsilon, tau0)) * (divergence - dfphi);
        }
    }
}

__global__ void boundaryConditionsPhi(double *phi) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x == 0 || x == DATAXSIZE - 1 || y == 0 || y == DATAYSIZE - 1 || z == 0 || z == DATAZSIZE - 1) {
            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            phi[idx] = -1.0;
        }
    }
}

__global__ void thermalEquation(double *unew, double *uold, double *phinew, double *phiold, double D, double dt, double dx, double dy, double dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z