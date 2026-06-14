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
  if (idx == x && idy == y && idz == z) {
    *phix = (phi[(x+1)%DATAXSIZE][y][z] - phi[(x-1+DATAXSIZE)%DATAXSIZE][y][z]) / (2.0*dx);
  }
}

__global__ void GradientY(double *phi, double dx, double dy, double dz, int x, int y, int z, double *phiy)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z) {
    *phiy = (phi[x][(y+1)%DATAYSIZE][z] - phi[x][(y-1+DATAYSIZE)%DATAYSIZE][z]) / (2.0*dy);
  }
}

__global__ void GradientZ(double *phi, double dx, double dy, double dz, int x, int y, int z, double *phiz)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z) {
    *phiz = (phi[x][y][(z+1)%DATAZSIZE] - phi[x][y][(z-1+DATAZSIZE)%DATAZSIZE]) / (2.0*dz);
  }
}

__global__ void Divergence(double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, int x, int y, int z, double *div)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z) {
    double phix, phiy, phiz;
    GradientX<<<1, 1>>>(Fx, dx, dy, dz, x, y, z, &phix);
    GradientY<<<1, 1>>>(Fy, dx, dy, dz, x, y, z, &phiy);
    GradientZ<<<1, 1>>>(Fz, dx, dy, dz, x, y, z, &phiz);
    *div = phix + phiy + phiz;
  }
}

__global__ void Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z, double *lap)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == x && idy == y && idz == z) {
    double phixx, phiyy, phizz;
    GradientX<<<1, 1>>>(phi, dx, dy, dz, (x+1)%DATAXSIZE, y, z, &phixx);
    GradientX<<<1, 1>>>(phi, dx, dy, dz, (x-1+DATAXSIZE)%DATAXSIZE, y, z, &phixx);
    GradientY<<<1, 1>>>(phi, dx, dy, dz, x, (y+1)%DATAYSIZE, z, &phiyy);
    GradientY<<<1, 1>>>(phi, dx, dy, dz, x, (y-1+DATAYSIZE)%DATAYSIZE, z, &phiyy);
    GradientZ<<<1, 1>>>(phi, dx, dy, dz, x, y, (z+1)%DATAZSIZE, &phizz);
    GradientZ<<<1, 1>>>(phi, dx, dy, dz, x, y, (z-1+DATAZSIZE)%DATAZSIZE, &phizz);
    *lap = (phixx + phiyy + phizz) / (SQ(dx) + SQ(dy) + SQ(dz));
  }
}

__global__ void An(double phix, double phiy, double phiz, double epsilon, double *an)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == 0 && idy == 0 && idz == 0) {
    if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
      *an = ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
             ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
             ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
    }
    else
    {
      *an = (1.0-((5.0/3.0)*epsilon));
    }
  }
}

__global__ void Wn(double phix, double phiy, double phiz, double epsilon, double W0, double *wn)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == 0 && idy == 0 && idz == 0) {
    *wn = (W0*An(phix,phiy,phiz,epsilon));
  }
}

__global__ void taun(double phix, double phiy, double phiz, double epsilon, double tau0, double *taun)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == 0 && idy == 0 && idz == 0) {
    *taun = tau0 * SQ(An(phix,phiy,phiz,epsilon));
  }
}

__global__ void dFunc(double l, double m, double n, double *dfunc)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx == 0 && idy == 0 && idz == 0) {
    if (l != 0.0 || m != 0.0 || n != 0.0){
      *dfunc = (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
    }
    else
    {
      *dfunc = 0.0;
    }
  }
}

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz,
                               double dx, double dy, double dz, double epsilon, double W0, double tau0)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx < DATAXSIZE && idy < DATAYSIZE && idz < DATAZSIZE) {
    if ((idx < (DATAXSIZE-1)) && (idy < (DATAYSIZE-1)) && 
        (idz < (DATAZSIZE-1)) && (idx > (0)) && 
        (idy > (0)) && (idz > (0))) {

      double phix, phiy, phiz;
      GradientX<<<1, 1>>>(phi, dx, dy, dz, idx, idy, idz, &phix);
      GradientY<<<1, 1>>>(phi, dx, dy, dz, idx, idy, idz, &phiy);
      GradientZ<<<1, 1>>>(phi, dx, dy, dz, idx, idy, idz, &phiz);
      double sqGphi = SQ(phix