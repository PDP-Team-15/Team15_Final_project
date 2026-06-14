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

double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

__global__ void GradientX(double *phi, double dx, double dy, double dz, int x, int y, int z, double *phix)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z)
  {
    phix[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = (phi[(idx+1) + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] - phi[(idx-1) + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) / (2.0*dx);
  }
}

__global__ void GradientY(double *phi, double dx, double dy, double dz, int x, int y, int z, double *phiy)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z)
  {
    phiy[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = (phi[idx + (idy+1) * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] - phi[idx + (idy-1) * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE]) / (2.0*dy);
  }
}

__global__ void GradientZ(double *phi, double dx, double dy, double dz, int x, int y, int z, double *phiz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z)
  {
    phiz[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = (phi[idx + idy * DATAXSIZE + (idz+1) * DATAYSIZE * DATAXSIZE] - phi[idx + idy * DATAXSIZE + (idz-1) * DATAYSIZE * DATAXSIZE]) / (2.0*dz);
  }
}

__global__ void Divergence(double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, int x, int y, int z, double *div)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z)
  {
    div[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] = GradientX(Fx, dx, dy, dz, x, y, z, NULL)[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] + 
    GradientY(Fy, dx, dy, dz, x, y, z, NULL)[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE] + 
    GradientZ(Fz, dx, dy, dz, x, y, z, NULL)[idx + idy * DATAXSIZE + idz * DATAYSIZE * DATAXSIZE];
  }
}

__global__ void Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z, double *lap)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z)
  {
    double phixx = (phi[(idx+1) + idy *