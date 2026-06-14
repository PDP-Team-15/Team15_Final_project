```cuda
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
#include <string.h>
#include "reference.h"
#endif

__global__ void dFphi(double *phi, double *u, double *lambda, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  double phix = GradientX(phi, u, lambda, i, j, k, nx, ny, nz);
  double phiy = GradientY(phi, u, lambda, i, j, k, nx, ny, nz);
  double phiz = GradientZ(phi, u, lambda, i, j, k, nx, ny, nz);

  phi[i + nx * (j + ny * k)] = (-phi[i + nx * (j + ny * k)]*(1.0-phi[i + nx * (j + ny * k)]*phi[i + nx * (j + ny * k)])+lambda[i + nx * (j + ny * k)]*u[i + nx * (j + ny * k)]*(1.0-phi[i + nx * (j + ny * k)]*phi[i + nx * (j + ny * k)])*(1.0-phi[i + nx * (j + ny * k)]*phi[i + nx * (j + ny * k)]));
}

__global__ void GradientX(double *phi, double *u, double *lambda, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  phi[i + nx * (j + ny * k)] = (phi[(i+1) + nx * (j + ny * k)] - phi[(i-1) + nx * (j + ny * k)]) / (2.0);
}

__global__ void GradientY(double *phi, double *u, double *lambda, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  phi[i + nx * (j + ny * k)] = (phi[i + nx * (j+1 + ny * k)] - phi[i + nx * (j-1 + ny * k)]) / (2.0);
}

__global__ void GradientZ(double *phi, double *u, double *lambda, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  phi[i + nx * (j + ny * k)] = (phi[i + nx * (j + ny * (k+1))] - phi[i + nx * (j + ny * (k-1))]) / (2.0);
}

__global__ void Divergence(double *Fx, double *Fy, double *Fz, double *phi, double dx, double dy, double dz, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  Fx[i + nx * (j + ny * k)] = GradientX(Fx, phi, dx, dy, dz, i, j, k, nx, ny, nz);
  Fy[i + nx * (j + ny * k)] = GradientY(Fy, phi, dx, dy, dz, i, j, k, nx, ny, nz);
  Fz[i + nx * (j + ny * k)] = GradientZ(Fz, phi, dx, dy, dz, i, j, k, nx, ny, nz);

  Fx[i + nx * (j + ny * k)] += Fy[i + nx * (j + ny * k)] + Fz[i + nx * (j + ny * k)];
}

__global__ void Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  double phixx = (phi[(i+1) + nx * (j + ny * k)] + phi[(i-1) + nx * (j + ny * k)] - 2.0 * phi[i + nx * (j + ny * k)]) / SQ(dx);
  double phiyy = (phi[i + nx * (j+1 + ny * k)] + phi[i + nx * (j-1 + ny * k)] - 2.0 * phi[i + nx * (j + ny * k)]) / SQ(dy);
  double phizz = (phi[i + nx * (j + ny * (k+1))] + phi[i + nx * (j + ny * (k-1))] - 2.0 * phi[i + nx * (j + ny * k)]) / SQ(dz);

  phi[i + nx * (j + ny * k)] = phixx + phiyy + phizz;
}

__global__ void An(double *phix, double *phiy, double *phiz, double *epsilon, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  if (phix[i + nx * (j + ny * k)] != 0.0 || phiy[i + nx * (j + ny * k)] != 0.0 || phiz[i + nx * (j + ny * k)] != 0.0){
    phix[i + nx * (j + ny * k)] = ((1.0 - 3.0 * epsilon[i + nx * (j + ny * k)]) * (1.0 + (((4.0 * epsilon[i + nx * (j + ny * k)]) / (1.0-3.0*epsilon[i + nx * (j + ny * k)])))*((SQ(phix[i + nx * (j + ny * k)])*SQ(phix[i + nx * (j + ny * k)])+SQ(phiy[i + nx * (j + ny * k)])*SQ(phiy[i + nx * (j + ny * k)])+SQ(phiz[i + nx * (j + ny * k)])*SQ(phiz[i + nx * (j + ny * k)])) / ((SQ(phix[i + nx * (j + ny * k)])+SQ(phiy[i + nx * (j + ny * k)])+SQ(phiz[i + nx * (j + ny * k)]))*(SQ(phix[i + nx * (j + ny * k)])+SQ(phiy[i + nx * (j + ny * k)])+SQ(phiz[i + nx * (j + ny * k)])))))));
  }
  else
  {
    phix[i + nx * (j + ny * k)] = (1.0-((5.0/3.0)*epsilon[i + nx * (j + ny * k)]));
  }
}

__global__ void Wn(double *phix, double *phiy, double *phiz, double *epsilon, double *W0, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  phix[i + nx * (j + ny * k)] = W0[i + nx * (j + ny * k)]*An(phix[i + nx * (j + ny * k)],phiy[i + nx * (j + ny * k)],phiz[i + nx * (j + ny * k)],epsilon[i + nx * (j + ny * k)]);
}

__global__ void taun(double *phix, double *phiy, double *phiz, double *epsilon, double *tau0, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  phix[i + nx * (j + ny * k)] = tau0[i + nx * (j + ny * k)] * SQ(An(phix[i + nx * (j + ny * k)],phiy[i + nx * (j + ny * k)],phiz[i + nx * (j + ny * k)],epsilon[i + nx * (j + ny * k)]));
}

__global__ void dFunc(double *l, double *m, double *n, int x, int y, int z, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  if (l[i + nx * (j + ny * k)] != 0.0 || m[i + nx * (j + ny * k)] != 0.0 || n[i + nx * (j + ny * k)] != 0.0){
    l[i + nx * (j + ny * k)] = (((l[i + nx * (j + ny * k)]*l[i + nx * (j + ny * k)]*l[i + nx * (j + ny * k)]*(SQ(m[i + nx * (j + ny * k)])+SQ(n[i + nx * (j + ny * k)])))-(l[i + nx * (j + ny * k)]*(SQ(m[i + nx * (j + ny * k)])*SQ(m[i + nx * (j + ny * k)])+SQ(n[i + nx * (j + ny * k)])*SQ(n[i + nx * (j + ny * k)])))) / ((SQ(l[i + nx * (j + ny * k)])+SQ(m[i + nx * (j + ny * k)])+SQ(n[i + nx * (j + ny * k)]))*(SQ(l[i + nx * (j + ny * k)])+SQ(m[i + nx * (j + ny * k)])+SQ(n[i + nx * (j + ny * k)]))));
  }
  else
  {
    l[i + nx * (j + ny * k)] = 0.0;
  }
}

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, double *epsilon, double *W0, double *tau0, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  if ((i < (nx-1)) && (j < (ny-1)) && (k < (nz-1)) && (i > (0)) && (j > (0)) && (k > (0))){
    double phix = GradientX(phi, dx, dy, dz, i, j, k, nx, ny, nz);
    double phiy = GradientY(phi, dx, dy, dz, i, j, k, nx, ny, nz);
    double phiz = GradientZ(phi, dx, dy, dz, i, j, k, nx, ny, nz);
    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
    double c = 16.0 * W0[i + nx * (j + ny * k)] * epsilon[i + nx * (j + ny * k)];
    double w = Wn(phix,phiy,phiz,epsilon,W0);
    double w2 = SQ(w);

    Fx[i + nx * (j + ny * k)] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz,epsilon,W0);
    Fy[i + nx * (j + ny * k)] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix,epsilon,W0);
    Fz[i + nx * (j + ny * k)] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy,epsilon,W0);
  }
  else
  {
    Fx[i + nx * (j + ny * k)] = 0.0;
    Fy[i + nx * (j + ny * k)] = 0.0;
    Fz[i + nx * (j + ny * k)] = 0.0;
  }
}

__global__ void allenCahn(double *phinew, double *phiold, double *uold, double *Fx, double *Fy, double *Fz, double *epsilon, double *W0, double *tau0, double *lambda, double dt, double dx, double dy, double dz, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  double phix = GradientX(phiold, dx, dy, dz, i, j, k, nx, ny, nz);
  double phiy = GradientY(phiold, dx, dy, dz, i, j, k, nx, ny, nz);
  double phiz = GradientZ(phiold, dx, dy, dz, i, j, k, nx, ny, nz);

  phinew[i + nx * (j + ny * k)] = phiold[i + nx * (j + ny * k)] + 
   (dt / taun(phix,phiy,phiz,epsilon,tau0, nx, ny, nz)) * 
   (Divergence(Fx,Fy,Fz,phiold, dx, dy, dz, i, j, k, nx, ny, nz) - 
    dFphi(phiold, uold, lambda, i, j, k, nx, ny, nz));
}

__global__ void boundaryConditionsPhi(double *phinew, int nx, int ny, int nz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= nx || idy >= ny || idz >= nz) return;

  int i = idx;
  int j = idy;
  int k = idz;

  if (i == 0){
    phinew[i + nx * (j + ny * k)] = -1.0;
  }
  else if (i == nx-1){
    phinew[i + nx * (j + ny * k)] = -1.0;
  }
  else if (j == 0){
    phinew[i + nx * (j + ny * k)] = -1.0;
  }
  else if (j == ny-1){
    phinew[i + nx * (j + ny * k)] = -1.0;
  }
  else if (k == 0){
    phinew[i + nx * (j + ny * k)] = -1.0;
