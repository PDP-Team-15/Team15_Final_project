#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400

#define SQ(x) ((x)*(x))

typedef double nRarray[DATAYSIZE][DATAXSIZE];

#ifdef VERIFY
#include "reference.h"
#endif

__global__ void dFphi(double *phi, double *u, double *lambda, int idx, int idy, int idz)
{
  int x = blockIdx.x * blockDim.x + idx;
  int y = blockIdx.y * blockDim.y + idy;
  int z = blockIdx.z * blockDim.z + idz;
  phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] = (-phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * (1.0 - phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + *lambda * *u * (1.0 - phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) * (1.0 - phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * phi[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]));
}

__global__ void GradientX(double *phi, double dx, double dy, double dz, int x, int y, int z)
{
  int idx = blockIdx.x * blockDim.x + x;
  int idy = blockIdx.y * blockDim.y + y;
  int idz = blockIdx.z * blockDim.z + z;
  phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = (phi[(idx + 1) + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] - phi[(idx - 1) + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) / (2.0 * dx);
}

__global__ void GradientY(double *phi, double dx, double dy, double dz, int x, int y, int z)
{
  int idx = blockIdx.x * blockDim.x + x;
  int idy = blockIdx.y * blockDim.y + y;
  int idz = blockIdx.z * blockDim.z + z;
  phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = (phi[idx + (idy + 1) * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] - phi[idx + (idy - 1) * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) / (2.0 * dy);
}

__global__ void GradientZ(double *phi, double dx, double dy, double dz, int x, int y, int z)
{
  int idx = blockIdx.x * blockDim.x + x;
  int idy = blockIdx.y * blockDim.y + y;
  int idz = blockIdx.z * blockDim.z + z;
  phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = (phi[idx + idy * DATAXSIZE + (idz + 1) * DATAYSIZE * DATAXSIZE] - phi[idx + idy * DATAXSIZE + (idz - 1) * DATAYSIZE * DATAXSIZE]) / (2.0 * dz);
}

__global__ void Divergence(double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, int x, int y, int z)
{
  int idx = blockIdx.x * blockDim.x + x;
  int idy = blockIdx.y * blockDim.y + y;
  int idz = blockIdx.z * blockDim.z + z;
  Fx[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = GradientX(Fx, dx, dy, dz, idx, idy, idz);
  Fy[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = GradientY(Fy, dx, dy, dz, idx, idy, idz);
  Fz[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = GradientZ(Fz, dx, dy, dz, idx, idy, idz);
}

__global__ void Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z)
{
  int idx = blockIdx.x * blockDim.x + x;
  int idy = blockIdx.y * blockDim.y + y;
  int idz = blockIdx.z * blockDim.z + z;
  double phixx = (phi[(idx + 1) + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] + phi[(idx - 1) + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] - 2.0 * phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) / SQ(dx);
  double phiyy = (phi[idx + (idy + 1) * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] + phi[idx + (idy - 1) * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] - 2.0 * phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) / SQ(dy);
  double phizz = (phi[idx + idy * DATAXSIZE + (idz + 1) * DATAYSIZE * DATAXSIZE] + phi[idx + idy * DATAXSIZE + (idz - 1) * DATAYSIZE * DATAXSIZE] - 2.0 * phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) / SQ(dz);
  phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = phixx + phiyy + phizz;
}

__global__ void An(double *phix, double *phiy, double *phiz, double epsilon, int idx, int idy, int idz)
{
  int x = blockIdx.x * blockDim.x + idx;
  int y = blockIdx.y * blockDim.y + idy;
  int z = blockIdx.z * blockDim.z + idz;
  if (phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] != 0.0 || phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] != 0.0 || phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] != 0.0) {
    phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] = ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0 - 3.0 * epsilon)) * ((SQ(phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE])) / ((SQ(phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE])) * (SQ(phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]))))));
  }
  else {
    phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] = (1.0 - ((5.0 / 3.0) * epsilon));
  }
}

__global__ void Wn(double *phix, double *phiy, double *phiz, double epsilon, double W0, int idx, int idy, int idz)
{
  int x = blockIdx.x * blockDim.x + idx;
  int y = blockIdx.y * blockDim.y + idy;
  int z = blockIdx.z * blockDim.z + idz;
  phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] = W0 * An(phix, phiy, phiz, epsilon, x, y, z);
}

__global__ void taun(double *phix, double *phiy, double *phiz, double epsilon, double tau0, int idx, int idy, int idz)
{
  int x = blockIdx.x * blockDim.x + idx;
  int y = blockIdx.y * blockDim.y + idy;
  int z = blockIdx.z * blockDim.z + idz;
  phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] = tau0 * SQ(An(phix, phiy, phiz, epsilon, x, y, z));
}

__global__ void dFunc(double *phix, double *phiy, double *phiz, int idx, int idy, int idz)
{
  int x = blockIdx.x * blockDim.x + idx;
  int y = blockIdx.y * blockDim.y + idy;
  int z = blockIdx.z * blockDim.z + idz;
  if (phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] != 0.0 || phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] != 0.0 || phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] != 0.0) {
    phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] = (((phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * (SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE])) - phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] * (SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) * SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) * SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]))) / ((SQ(phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE])) * (SQ(phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiy[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]) + SQ(phiz[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE]))));
  }
  else {
    phix[x + y * DATAXSIZE + z * DATAYSIZE * DATAXSIZE] = 0.0;
  }
}

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, double epsilon, double W0, double tau0)
{
  int idx = blockIdx.x * blockDim.x;
  int idy = blockIdx.y * blockDim.y;
  int idz = blockIdx.z * blockDim.z;
  if ((idx < DATAXSIZE - 1) && (idy < DATAYSIZE - 1) && (idz < DATAZSIZE - 1) && (idx > 0) && (idy > 0) && (idz > 0)) {
    double phix = GradientX(phi, dx, dy, dz, idx, idy, idz);
    double phiy = GradientY(phi, dx, dy, dz, idx, idy, idz);
    double phiz = GradientZ(phi, dx, dy, dz, idx, idy, idz);
    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
    double c = 16.0 * W0 * epsilon;
    double w = Wn(phix, phiy, phiz, epsilon, W0, idx, idy, idz);
    double w2 = SQ(w);
    Fx[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = w2 * phix + sqGphi * w * c * dFunc(phix, phiy, phiz, idx, idy, idz);
    Fy[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = w2 * phiy + sqGphi * w * c * dFunc(phiy, phiz, phix, idx, idy, idz);
    Fz[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = w2 * phiz + sqGphi * w * c * dFunc(phiz, phix, phiy, idx, idy, idz);
  }
  else {
    Fx[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = 0.0;
    Fy[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = 0.0;
    Fz[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = 0.0;
  }
}

__global__ void allenCahn(double *phinew, double *phiold, double *uold, double *Fx, double *Fy, double *Fz, double epsilon, double W0, double tau0, double lambda, double dt, double dx, double dy, double dz)
{
  int idx = blockIdx.x * blockDim.x;
  int idy = blockIdx.y * blockDim.y;
  int idz = blockIdx.z * blockDim.z;
  if ((idx > 0) && (idx < DATAXSIZE - 1) && (idy > 0) && (idy < DATAYSIZE - 1) && (idz > 0) && (idz < DATAZSIZE - 1)) {
    double phix = GradientX(phiold, dx, dy, dz, idx, idy, idz);
    double phiy = GradientY(phiold, dx, dy, dz, idx, idy, idz);
    double phiz = GradientZ(phiold, dx, dy, dz, idx, idy, idz);
    phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = phiold[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] + (dt / taun(phix, phiy, phiz, epsilon, tau0, idx, idy, idz)) * (Divergence(Fx, Fy, Fz, dx, dy, dz, idx, idy, idz) - dFphi(phiold, uold, &lambda, idx, idy, idz));
  }
}

__global__ void boundaryConditionsPhi(double *phinew)
{
  int idx = blockIdx.x * blockDim.x;
  int idy = blockIdx.y * blockDim.y;
  int idz = blockIdx.z * blockDim.z;
  if (idx == 0) {
    phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -1.0;
  }
  else if (idx == DATAXSIZE - 1) {
    phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -1.0;
  }
  else if (idy == 0) {
    phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -1.0;
  }
  else if (idy == DATAYSIZE - 1) {
    phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -1.0;
  }
  else if (idz == 0) {
    phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -1.0;
  }
  else if (idz == DATAZSIZE - 1) {
    phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -1.0;
  }
}

__global__ void thermalEquation(double *unew, double *uold, double *phinew, double *phiold, double D, double dt, double dx, double dy, double dz)
{
  int idx = blockIdx.x * blockDim.x;
  int idy = blockIdx.y * blockDim.y;
  int idz = blockIdx.z * blockDim.z;
  if ((idx > 0) && (idx < DATAXSIZE - 1) && (idy > 0) && (idy < DATAYSIZE - 1) && (idz > 0) && (idz < DATAZSIZE - 1)) {
    unew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = uold[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] + 0.5 * (phinew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] - phiold[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) + dt * D * Laplacian(uold, dx, dy, dz, idx, idy, idz);
  }
}

__global__ void boundaryConditionsU(double *unew, double delta)
{
  int idx = blockIdx.x * blockDim.x;
  int idy = blockIdx.y * blockDim.y;
  int idz = blockIdx.z * blockDim.z;
  if (idx == 0) {
    unew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -delta;
  }
  else if (idx == DATAXSIZE - 1) {
    unew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -delta;
  }
  else if (idy == 0) {
    unew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -delta;
  }
  else if (idy == DATAYSIZE - 1) {
    unew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -delta;
  }
  else if (idz == 0) {
    unew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -delta;
  }
  else if (idz == DATAZSIZE - 1) {
    unew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -delta;
  }
}

__global__ void swapGrid(double *cnew, double *cold)
{
  int idx = blockIdx.x * blockDim.x;
  int idy = blockIdx.y * blockDim.y;
  int idz = blockIdx.z * blockDim.z;
  double tmp = cnew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE];
  cnew[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = cold[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE];
  cold[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = tmp;
}

void initializationPhi(double *phi, double r0)
{
  int idx, idy, idz;
  for (idx = 0; idx < DATAXSIZE; idx++) {
    for (idy = 0; idy < DATAYSIZE; idy++) {
      for (idz = 0; idz < DATAZSIZE; idz++) {
        double r = sqrt(SQ(idx - 0.5 * DATAXSIZE) + SQ(idy - 0.5 * DATAYSIZE) + SQ(idz - 0.5 * DATAZSIZE));
        if (r < r0) {
          phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = 1.0;
        }
        else {
          phi[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -1.0;
        }
      }
    }
  }
}

void initializationU(double *u, double r0, double delta)
{
  int idx, idy, idz;
  for (idx = 0; idx < DATAXSIZE; idx++) {
    for (idy = 0; idy < DATAYSIZE; idy++) {
      for (idz = 0; idz < DATAZSIZE; idz++) {
        double r = sqrt(SQ(idx - 0.5 * DATAXSIZE) + SQ(idy - 0.5 * DATAYSIZE) + SQ(idz - 0.5 * DATAZSIZE));
        if (r < r0) {
          u[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = 0.0;
        }
        else {
          u[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = -delta * (1.0 - exp(-(r - r0)));
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
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
  const double lambda = (W0 * a1) / (d0);
  const double tau0 = ((W0 * W0 * W0 * a1 * a2) / (d0 * D)) + ((W0 * W0 * beta0) / (d0));

  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_in_bytes = sizeof(double) * vol;

  double *phi_host = (double*)malloc(vol_in_bytes);
  double *u_host = (double*)malloc(vol_in_bytes);
  initializationPhi(phi_host, r0);
  initializationU(u_host, r0, delta);

#ifdef VERIFY
  double *phi_ref = (double*)malloc(vol_in_bytes);
  double *u_ref = (double*)malloc(vol_in_bytes);
  memcpy(phi_ref, phi_host, vol_in_bytes);
  memcpy(u_ref, u_host, vol_in_bytes);
  reference(phi_ref, u_ref, vol, num_steps);
#endif

  double *d_phiold;
  double *d_uold;
  double *d_phinew;
  double *d_unew;
  double *d_Fx;
  double *d_Fy;
  double *d_Fz;

  cudaMalloc((void**)&d_phiold, vol_in_bytes);
  cudaMalloc((void**)&d_uold, vol_in_bytes);
  cudaMalloc((void**)&d_phinew, vol_in_bytes);
  cudaMalloc((void**)&d_unew, vol_in_bytes);
  cudaMalloc((void**)&d_Fx, vol_in_bytes);
  cudaMalloc((void**)&d_Fy, vol_in_bytes);
  cudaMalloc((void**)&d_Fz, vol_in_bytes);

  cudaMemcpy(d_phiold, phi_host, vol_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uold, u_host, vol_in_bytes, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocksX = (nx + blockSize - 1) / blockSize;
  int numBlocksY = (ny + blockSize - 1) / blockSize;
  int numBlocksZ = (nz + blockSize - 1) / blockSize;

  for (int t = 0; t <= num_steps; t++) {
    calculateForce<<<dim3(numBlocksX, numBlocksY, numBlocksZ), blockSize>>>(d_phiold, d_Fx, d_Fy, d_Fz, dx, dy, dz, epsilon, W0, tau0);
    allenCahn<<<dim3(numBlocksX, numBlocksY, numBlocksZ), blockSize>>>(d_phinew, d_phiold, d_uold, d_Fx, d_Fy, d_Fz, epsilon, W0, tau0, lambda, dt, dx, dy, dz);
    boundaryConditionsPhi<<<dim3(numBlocksX, numBlocksY, numBlocksZ), blockSize>>>(d_phinew);
    thermalEquation<<<dim3(numBlocksX, numBlocksY, numBlocksZ), blockSize>>>(d_unew, d_uold, d_phinew, d_phiold, D, dt, dx, dy, dz);
    boundaryConditionsU<<<dim3(numBlocksX, numBlocksY, numBlocksZ), blockSize>>>(d_unew, delta);
    swapGrid<<<dim3(numBlocksX, numBlocksY, numBlocksZ), blockSize>>>(d_phinew, d_phiold);
    swapGrid<<<dim3(numBlocksX, numBlocksY, numBlocksZ), blockSize>>>(d_unew, d_uold);
  }

  cudaMemcpy(phi_host, d_phiold, vol_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(u_host, d_uold, vol_in_bytes, cudaMemcpyDeviceToHost);

#ifdef VERIFY
  bool ok = true;
  for (int idx = 0; idx < nx; idx++)
    for (int idy = 0; idy < ny; idy++)
      for (int idz = 0; idz < nz; idz++) {
        if (fabs(phi_ref[idx + idy * nx + idz * ny * nx] - phi_host[idx + idy * nx + idz * ny * nx]) > 1e-3) {
          ok = false; printf("phi: %lf %lf\n", phi_ref[idx + idy * nx + idz * ny * nx], phi_host[idx + idy * nx + idz * ny * nx]);
	}
        if (fabs(u_ref[idx + idy * nx + idz * ny * nx] - u_host[idx + idy * nx + idz * ny * nx]) > 1e-3) {
          ok = false; printf("u: %lf %lf\n", u_ref[idx + idy * nx + idz * ny * nx], u_host[idx + idy * nx + idz * ny * nx]);
        }
      }
  printf("%s\n", ok ? "PASS" : "FAIL");
  free(phi_ref);
  free(u_ref);
#endif

  free(phi_host);
  free(u_host);
  cudaFree(d_phiold);
  cudaFree(d_uold);
  cudaFree(d_phinew);
  cudaFree(d_unew);
  cudaFree(d_Fx);
  cudaFree(d_Fy);
  cudaFree(d_Fz);
  return 0;
}