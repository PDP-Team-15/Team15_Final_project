Here is the CUDA translation of your OpenMP code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

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

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz,
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
    

    Fx[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
    Fy[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
    Fz[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
  }
  else
  {
    Fx[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = 0.0;
    Fy[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = 0.0;
    Fz[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = 0.0;
  }
}

__global__ void allenCahn(double *phinew, double *phiold, double *uold,
                         double *Fx, double *Fy, double *Fz,
                         double epsilon, double W0, double tau0, double lambda,
                         double dt, double dx, double dy, double dz)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix > 0 && ix < DATAXSIZE-1 && iy > 0 && iy < DATAYSIZE-1 && iz > 0 && iz < DATAZSIZE-1) {

    double phix = GradientX(phiold,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phiold,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phiold,dx,dy,dz,ix,iy,iz); 

    phinew[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = phiold[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] + 
     (dt / taun(phix,phiy,phiz,epsilon,tau0)) * 
     (Divergence(Fx+Fx[iy*DATAXSIZE + iz],Fy+Fy[iy*DATAXSIZE + iz],Fz+Fz[iy*DATAXSIZE + iz],dx,dy,dz,ix,iy,iz) - 
      dFphi(phiold[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz], uold[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz],lambda));
  }
}

__global__ void boundaryConditionsPhi(double *phinew)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix >= 0 && ix < DATAXSIZE && iy >= 0 && iy < DATAYSIZE && iz >= 0 && iz < DATAZSIZE) {

    if (ix == 0){
      phinew[ix*DATAYSIZE*DATAXSIZE + iy*DATAXSIZE + iz] = -1.0;
    }
    else if (ix ==