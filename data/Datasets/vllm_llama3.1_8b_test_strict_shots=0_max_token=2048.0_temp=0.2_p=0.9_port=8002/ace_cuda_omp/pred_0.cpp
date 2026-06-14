```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

double GradientX(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);
}

double GradientY(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);
}

double GradientZ(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);
}

double Divergence(double phix[][DATAYSIZE][DATAXSIZE], 
                  double phiy[][DATAYSIZE][DATAXSIZE],
                  double phiz[][DATAYSIZE][DATAXSIZE], 
                  double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) + 
         GradientY(phiy,dx,dy,dz,x,y,z) +
         GradientZ(phiz,dx,dy,dz,x,y,z);
}

double Laplacian(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0 * phi[x][y][z]) / SQ(dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0 * phi[x][y][z]) / SQ(dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0 * phi[x][y][z]) / SQ(dz);
  return phixx + phiyy + phizz;
}

double An(double phix, double phiy, double phiz, double epsilon)
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

double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An(phix,phiy,phiz,epsilon));
}

double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An(phix,phiy,phiz,epsilon));
}

double dFunc(double l, double m, double n)
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

void calculateForce(double phi[][DATAYSIZE][DATAXSIZE], 
                    double Fx[][DATAYSIZE][DATAXSIZE],
                    double Fy[][DATAYSIZE][DATAXSIZE],
                    double Fz[][DATAYSIZE][DATAXSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0)
{
  #pragma omp parallel for collapse(3)
  for (int iz = 0; iz < DATAZSIZE; iz++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int ix = 0; ix < DATAXSIZE; ix++) {
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
    }
  }
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
  #pragma omp parallel for collapse(3)
  for (int iz = 0; iz < DATAZSIZE; iz++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int ix = 0; ix < DATAXSIZE; ix++) {
        if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
            (iz < (DATAZSIZE-1)) && (ix > (0)) && 
            (iy > (0)) && (iz > (0))) {

          double phix = GradientX(phiold,dx,dy,dz,ix,iy,iz);
          double phiy = GradientY(phiold,dx,dy,dz,ix,iy,iz);
          double phiz = GradientZ(phiold,dx,dy,dz,ix,iy,iz); 

          phinew[ix][iy][iz] = phiold[ix][iy][iz] + 
           (dt / taun(phix,phiy,phiz,epsilon,tau0)) * 
           (Divergence(Fx,Fy,Fz,dx,dy,dz,ix,iy,iz) - 
            dFphi(phiold[ix][iy][iz], uold[ix][iy][iz],lambda));
        }
      }
    }
  }
}

void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAXSIZE])
{
  #pragma omp parallel for collapse(3)
  for (int iz = 0; iz < DATAZSIZE; iz++) {
    for (int iy = 0; iy < DATAYSIZE; iy++) {
      for (int ix = 0; ix < DATAXSIZE; ix++) {
        if (ix == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (ix == DATAXSIZE-1){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == 0){
          phinew[ix][iy][iz] = -1.0;
        }
        else if (iy == DATAYSIZE-1){
          phinew[ix