```cpp
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

__global__ void dFphi(double *phi, double *u, double *lambda, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double phix = GradientX(phi, idx, idy, idz, nx, ny, nz);
    double phiy = GradientY(phi, idx, idy, idz, nx, ny, nz);
    double phiz = GradientZ(phi, idx, idy, idz, nx, ny, nz);
    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
    double c = 16.0 * W0 * epsilon;
    double w = Wn(phix, phiy, phiz, epsilon, W0);
    double w2 = SQ(w);

    Fx[idx][idy][idz] = w2 * phix + sqGphi * w * c * dFunc(phix, phiy, phiz);
    Fy[idx][idy][idz] = w2 * phiy + sqGphi * w * c * dFunc(phiy, phiz, phix);
    Fz[idx][idy][idz] = w2 * phiz + sqGphi * w * c * dFunc(phiz, phix, phiy);
  }
}

__global__ void GradientX(double *phi, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double phix = (phi[idx+1][idy][idz] - phi[idx-1][idy][idz]) / (2.0*dx);
    phi[idx][idy][idz] = phix;
  }
}

__global__ void GradientY(double *phi, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double phiy = (phi[idx][idy+1][idz] - phi[idx][idy-1][idz]) / (2.0*dy);
    phi[idx][idy][idz] = phiy;
  }
}

__global__ void GradientZ(double *phi, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double phiz = (phi[idx][idy][idz+1] - phi[idx][idy][idz-1]) / (2.0*dz);
    phi[idx][idy][idz] = phiz;
  }
}

__global__ void Divergence(double *Fx, double *Fy, double *Fz, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double phix = GradientX(Fx, idx, idy, idz, nx, ny, nz);
    double phiy = GradientY(Fy, idx, idy, idz, nx, ny, nz);
    double phiz = GradientZ(Fz, idx, idy, idz, nx, ny, nz);
    double div = phix + phiy + phiz;
    Fx[idx][idy][idz] = div;
  }
}

__global__ void Laplacian(double *phi, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double phixx = (phi[idx+1][idy][idz] + phi[idx-1][idy][idz] - 2.0 * phi[idx][idy][idz]) / SQ(dx);
    double phiyy = (phi[idx][idy+1][idz] + phi[idx][idy-1][idz] - 2.0 * phi[idx][idy][idz]) / SQ(dy);
    double phizz = (phi[idx][idy][idz+1] + phi[idx][idy][idz-1] - 2.0 * phi[idx][idy][idz]) / SQ(dz);
    double laplacian = phixx + phiyy + phizz;
    phi[idx][idy][idz] = laplacian;
  }
}

__global__ void An(double *phix, double *phiy, double *phiz, double *epsilon, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double an = ((1.0 - 3.0 * epsilon[idx][idy][idz]) * (1.0 + (((4.0 * epsilon[idx][idy][idz]) / (1.0-3.0*epsilon[idx][idy][idz]))*
           ((SQ(phix[idx][idy][idz])*SQ(phix[idx][idy][idz])+SQ(phiy[idx][idy][idz])*SQ(phiy[idx][idy][idz])+SQ(phiz[idx][idy][idz])*SQ(phiz[idx][idy][idz])) /
           ((SQ(phix[idx][idy][idz])+SQ(phiy[idx][idy][idz])+SQ(phiz[idx][idy][idz]))*(SQ(phix[idx][idy][idz])+SQ(phiy[idx][idy][idz])+SQ(phiz[idx][idy][idz]))))));
    phix[idx][idy][idz] = an;
  }
}

__global__ void Wn(double *phix, double *phiy, double *phiz, double *epsilon, double *W0, int nx, int ny, int nz, int idx, int idy, int idz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
    int idx = i;
    int idy = j;
    int idz = k;
    double wn = (W0[idx][idy][idz]*An(phix[idx][idy][idz],phiy[idx][idy][idz],phiz[idx