Below is the translated CUDA version of the provided OpenMP code. The main function remains unchanged, and all function names match the original code.

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
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

__device__ double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

__device__ double GradientX(double phi[][DATAYSIZE][DATAXSIZE], 
                            double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);
}

__device__ double GradientY(double phi[][DATAYSIZE][DATAXSIZE], 
                            double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);
}

__device__ double GradientZ(double phi[][DATAYSIZE][DATAXSIZE], 
                            double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);
}

__device__ double Divergence(double phix[][DATAYSIZE][DATAXSIZE], 
                             double phiy[][DATAYSIZE][DATAXSIZE],
                             double phiz[][DATAYSIZE][DATAXSIZE], 
                             double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) + 
         GradientY(phiy,dx,dy,dz,x,y,z) +
         GradientZ(phiz,dx,dy,dz,x,y,z);
}

__device__ double Laplacian(double phi[][DATAYSIZE][DATAXSIZE],
                            double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0 * phi[x][y][z]) / SQ(dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0 * phi[x][y][z]) / SQ(dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0 * phi[x][y][z]) / SQ(dz);
  return phixx + phiyy + phizz;
}

__device__ double An(double phix, double phiy, double phiz, double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
    return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
           ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
           ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
  }
  else
  {
    return (1.0-((5.0/3.0)*epsilon));
  }
}

__device__ double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An(phix,phiy,phiz,epsilon));
}

__device__ double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An(phix,phiy,phiz,epsilon));
}

__device__ double dFunc(double l, double m, double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0){
    return (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
  }
  else
  {
    return 0.0;
  }
}

__global__ void calculateForceKernel(double phi[][DATAYSIZE][DATAXSIZE], 
                                    double Fx[][DATAYSIZE][DATAXSIZE],
                                    double Fy[][DATAYSIZE][DATAXSIZE],
                                    double Fz[][DATAYSIZE][DATAXSIZE],
                                    double dx, double dy, double dz,
                                    double epsilon, double W0, double tau0)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
      (iz < (DATAZSIZE-1)) && (ix > (0)) && 
      (iy > (0)) && (iz > (0))) {

    double phix = GradientX(phi,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phi,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phi,dx,dy,dz,ix,iy,iz);
    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
    double c = 16.0 * W0 * epsilon;
    double w = Wn(phix,phiy,phiz,epsilon,W0);
    double w2 = SQ(w);
    

    Fx[ix][iy][iz] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
    Fy[ix][iy][iz] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
    Fz[ix][iy][iz] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
  }
  else
  {
    Fx[ix][iy][iz] = 0.0;
    Fy[ix][iy][iz] = 0.0;
    Fz[ix][iy][iz] = 0.0;
  }
}

__global__ void allenCahnKernel(double phinew[][DATAYSIZE][DATAXSIZE], 
                               double phiold[][DATAYSIZE][DATAXSIZE],
                               double uold[][DATAYSIZE][DATAXSIZE],
                               double Fx[][DATAYSIZE][DATAXSIZE],
                               double Fy[][DATAYSIZE][DATAXSIZE],
                               double Fz[][DATAYSIZE][DATAXSIZE],
                               double epsilon, double W0, double tau0, double lambda,
                               double dt, double dx, double dy, double dz)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix >= 1 && ix < DATAXSIZE-1 && iy >= 1 && iy < DATAYSIZE-1 && iz >= 1 && iz < DATAZSIZE-1) {

    double phix = GradientX(phiold,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phiold,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phiold,dx,dy,dz,ix,iy,iz); 

    phinew[ix][iy][iz] = phiold[ix][iy][iz] + 
     (dt / taun(phix,phiy,phiz,epsilon,tau0)) * 
     (Divergence(Fx,Fy,Fz,dx,dy,dz,ix,iy,iz) - 
      dFphi(phiold[ix][iy][iz], uold[ix][iy][iz],lambda));
  }
}

__global__ void boundaryConditionsPhiKernel(double phinew[][DATAYSIZE][DATAXSIZE])
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix < DATAXSIZE && iy < DATAYSIZE && iz < DATAZSIZE) {

    if (ix == 0 || ix == DATAXSIZE-1 ||
        iy == 0 || iy == DATAYSIZE-1 ||
        iz == 0 || iz == DATAZSIZE-1){
      phinew[ix][iy][iz] = -1.0;
    }
  }
}

__global__ void thermalEquationKernel(double unew[][DATAYSIZE][DATAXSIZE],
                                     double uold[][DATAYSIZE][DATAXSIZE],
                                     double phinew[][DATAYSIZE][DATAXSIZE],
                                     double phiold[][DATAYSIZE][DATAXSIZE],
                                     double D, double dt, double dx, double dy, double dz)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix >= 1 && ix < DATAXSIZE-1 && iy >= 1 && iy < DATAYSIZE-1 && iz >= 1 && iz < DATAZSIZE-1) {

    unew[ix][iy][iz] = uold[ix][iy][iz] + 
      0.5*(phinew[ix][iy][iz]-
           phiold[ix][iy][iz]) +
      dt * D * Laplacian(uold,dx,dy,dz,ix,iy,iz);
  }
}

__global__ void boundaryConditionsUKernel(double unew[][DATAYSIZE][DATAXSIZE], double delta)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix < DATAXSIZE && iy < DATAYSIZE && iz < DATAZSIZE) {

    if (ix == 0 || ix == DATAXSIZE-1 ||
        iy == 0 || iy == DATAYSIZE-1 ||
        iz == 0 || iz == DATAZSIZE-1){
      unew[ix][iy][iz] =  -delta;
    }
  }
}

__global__ void swapGridKernel(double cnew[][DATAYSIZE][DATAXSIZE],
                              double cold[][DATAYSIZE][DATAXSIZE])
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix < DATAXSIZE && iy < DATAYSIZE && iz < DATAZSIZE) {
    double tmp = cnew[ix][iy][iz];
    cnew[ix][iy][iz] = cold[ix][iy][iz];
    cold[ix][iy][iz] = tmp;
  }
}

__global__ void initializationPhiKernel(double phi[][DATAYSIZE][DATAXSIZE], double r0)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix < DATAXSIZE && iy < DATAYSIZE && iz < DATAZSIZE) {
    double r = sqrt(SQ(ix-0.5*DATAXSIZE) + SQ(iy-0.5*DATAYSIZE) + SQ(iz-0.5*DATAZSIZE));
    if (r < r0){
      phi[ix][iy][iz] = 1.0;
    }
    else
    {
      phi[ix][iy][iz] = -1.0;
    }
  }
}

__global__ void initializationUKernel(double u[][DATAYSIZE][DATAXSIZE], double r0, double delta)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix < DATAXSIZE && iy < DATAYSIZE && iz < DATAZSIZE) {
    double r = sqrt(SQ(ix-0.5*DATAXSIZE) + SQ(iy-0.5*DATAYSIZE) + SQ(iz-0.5*DATAZSIZE));
    if (r < r0) {
      u[ix][iy][iz] = 0.0;
    }
    else
    {
      u[ix][iy][iz] = -delta * (1.0 - exp(-(r-r0)));
    }
  }
}

void calculateForce(double phi[][DATAYSIZE][DATAXSIZE], 
                    double Fx[][DATAYSIZE][DATAXSIZE],
                    double Fy[][DATAYSIZE][DATAXSIZE],
                    double Fz[][DATAYSIZE][DATAXSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0)
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  calculateForceKernel<<<numBlocks, threadsPerBlock>>>(phi, Fx, Fy, Fz, dx, dy, dz, epsilon, W0, tau0);
}

void allenCahn(double phinew[][DATAYSIZE][DATAXSIZE], 
               double phiold[][DATAYSIZE][DATAXSIZE],
               double uold[][DATAYSIZE][DATAXSIZE],
               double Fx[][DATAYSIZE][DATAXSIZE],
               double Fy[][DATAYSIZE][DATAXSIZE],
               double Fz[][DATAYSIZE][DATAXSIZE],
               double epsilon, double W0, double tau0, double lambda,
               double dt, double dx, double dy, double dz)
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  allenCahnKernel<<<numBlocks, threadsPerBlock>>>(phinew, phiold, uold, Fx, Fy, Fz, epsilon, W0, tau0, lambda, dt, dx, dy, dz);
}

void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAXSIZE])
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  boundaryConditionsPhiKernel<<<numBlocks, threadsPerBlock>>>(phinew);
}

void thermalEquation(double unew[][DATAYSIZE][DATAXSIZE],
                     double uold[][DATAYSIZE][DATAXSIZE],
                     double phinew[][DATAYSIZE][DATAXSIZE],
                     double phiold[][DATAYSIZE][DATAXSIZE],
                     double D, double dt, double dx, double dy, double dz)
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  thermalEquationKernel<<<numBlocks, threadsPerBlock>>>(unew, uold, phinew, phiold, D, dt, dx, dy, dz);
}

void boundaryConditionsU(double unew[][DATAYSIZE][DATAXSIZE], double delta)
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  boundaryConditionsUKernel<<<numBlocks, threadsPerBlock>>>(unew, delta);
}

void swapGrid(double cnew[][DATAYSIZE][DATAXSIZE],
              double cold[][DATAYSIZE][DATAXSIZE])
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  swapGridKernel<<<numBlocks, threadsPerBlock>>>(cnew, cold);
}

void initializationPhi(double phi[][DATAYSIZE][DATAXSIZE], double r0)
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  initializationPhiKernel<<<numBlocks, threadsPerBlock>>>(phi, r0);
}

void initializationU(double u[][DATAYSIZE][DATAXSIZE], double r0, double delta)
{
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((DATAXSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (DATAYSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (DATAZSIZE + threadsPerBlock.z - 1) / threadsPerBlock.z);
  initializationUK