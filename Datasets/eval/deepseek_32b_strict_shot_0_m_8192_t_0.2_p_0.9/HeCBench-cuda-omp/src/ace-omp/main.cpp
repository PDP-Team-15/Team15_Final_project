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
#include "omp.h"



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
  for (int x = 0; x < DATAXSIZE; x++) {
    for (int y = 0; y < DATAYSIZE; y++) {
      for (int z = 0; z < DATAZSIZE; z++) {
        if ((x < (DATAXSIZE-1)) && (y < (DATAYSIZE-1)) && 
            (z < (DATAZSIZE-1)) && (x > (0)) && 
            (y > (0)) && (z > (0))) {

          double phix = GradientX(phi,dx,dy,dz,x,y,z);
          double phiy = GradientY(phi,dx,dy,dz,x,y,z);
          double phiz = GradientZ(phi,dx,dy,dz,x,y,z);
          double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
          double c = 16.0 * W0 * epsilon;
          double w = Wn(phix,phiy,phiz,epsilon,W0);
          double w2 = SQ(w);

          Fx[x][y][z] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
          Fy[x][y][z] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
          Fz[x][y][z] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
        }
        else
        {
          Fx[x][y][z] = 0.0;
          Fy[x][y][z] = 0.0;
          Fz[x][y][z] = 0.0;
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
  for (int x = 0; x < DATAXSIZE; x++) {
    for (int y = 0; y < DATAYSIZE; y++) {
      for (int z = 0; z < DATAZSIZE; z++) {
        if ((x < (DATAXSIZE-1)) && (y < (DATAYSIZE-1)) && 
            (z < (DATAZSIZE-1)) && (x > (0)) && 
            (y > (0)) && (z > (0))) {

          double phix = GradientX(phiold,dx,dy,dz,x,y,z);
          double phiy = GradientY(phiold,dx,dy,dz,x,y,z);
          double phiz = GradientZ(phiold,dx,dy,dz,x,y,z); 

          phinew[x][y][z] = phiold[x][y][z] + 
           (dt / taun(phix,phiy,phiz,epsilon,tau0)) * 
           (Divergence(Fx,Fy,Fz,dx,dy,dz,x,y,z) - 
            dFphi(phiold[x][y][z], uold[x][y][z],lambda));
        }
      }
    }
  }
}

void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAXSIZE])
{
  #pragma omp parallel for collapse(3)
  for (int x = 0; x < DATAXSIZE; x++) {
    for (int y = 0; y < DATAYSIZE; y++) {
      for (int z = 0; z < DATAZSIZE; z++) {
        if (x == 0){
          phinew[x][y][z] = -1.0;
        }
        else if (x == DATAXSIZE-1){
          phinew[x][y][z] = -1.0;
        }
        else if (y == 0){
          phinew[x][y][z] = -1.0;
        }
        else if (y == DATAYSIZE-1){
          phinew[x][y][z] = -1.0;
        }
        else if (z == 0){
          phinew[x][y][z] = -1.0;
        }
        else if (z == DATAZSIZE-1){
          phinew[x][y][z] = -1.0;
        }
      }
    }
  }
}

void thermalEquation(double unew[][DATAYSIZE][DATAXSIZE],
                     double uold[][DATAYSIZE][DATAXSIZE],
                     double phinew[][DATAYSIZE][DATAXSIZE],
                     double phiold[][DATAYSIZE][DATAXSIZE],
                     double D, double dt, double dx, double dy, double dz)
{
  #pragma omp parallel for collapse(3)
  for (int x = 0; x < DATAXSIZE; x++) {
    for (int y = 0; y < DATAYSIZE; y++) {
      for (int z = 0; z < DATAZSIZE; z++) {
        if ((x < (DATAXSIZE-1)) && (y < (DATAYSIZE-1)) && 
            (z < (DATAZSIZE-1)) && (x > (0)) && 
            (y > (0)) && (z > (0))){
          unew[x][y][z] = uold[x][y][z] + 
            0.5*(phinew[x][y][z]- phiold[x][y][z]) +
            dt * D * Laplacian(uold,dx,dy,dz,x,y,z);
        }
      }
    }
  }
}

void boundaryConditionsU(double unew[][DATAYSIZE][DATAXSIZE], double delta)
{
  #pragma omp parallel for collapse(3)
  for (int x = 0; x < DATAXSIZE; x++) {
    for (int y = 0; y < DATAYSIZE; y++) {
      for (int z = 0; z < DATAZSIZE; z++) {
        if (x == 0){
          unew[x][y][z] =  -delta;
        }
        else if (x == DATAXSIZE-1){
          unew[x][y][z] =  -delta;
        }
        else if (y == 0){
          unew[x][y][z] =  -delta;
        }
        else if (y == DATAYSIZE-1){
          unew[x][y][z] =  -delta;
        }
        else if (z == 0){
          unew[x][y][z] =  -delta;
        }
        else if (z == DATAZSIZE-1){
          unew[x][y][z] =  -delta;
        }
      }
    }
  }
}

void swapGrid(double cnew[][DATAYSIZE][DATAXSIZE],
              double cold[][DATAYSIZE][DATAXSIZE])
{
  #pragma omp parallel for collapse(3)
  for (int x = 0; x < DATAXSIZE; x++) {
    for (int y = 0; y < DATAYSIZE; y++) {
      for (int z = 0; z < DATAZSIZE; z++) {
        double tmp = cnew[x][y][z];
        cnew[x][y][z] = cold[x][y][z];
        cold[x][y][z] = tmp;
      }
    }
  }
}

void initializationPhi(double phi[][DATAYSIZE][DATAXSIZE], double r0)
{
  #pragma omp parallel for collapse(3)
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0){
          phi[idx][idy][idz] = 1.0;
        }
        else
        {
          phi[idx][idy][idz] = -1.0;
        }
      }
    }
  }
}

void initializationU(double u[][DATAYSIZE][DATAXSIZE], double r0, double delta)
{
  #pragma omp parallel for collapse(3)
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0) {
          u[idx][idy][idz] = 0.0;
        }
        else
        {
          u[idx][idy][idz] = -delta * (1.0 - std::exp(-(r-r0)));
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
  const double a1 = 1.25 / std::sqrt(2.0);
  const double a2 = 0.64;
  const double lambda = (W0*a1)/(d0);
  const double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));

  

  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_in_bytes = sizeof(double) * vol;

  

  nRarray *phi_host = new nRarray[nx];
  nRarray *u_host = new nRarray[nx];

  initializationPhi(phi_host,r0);
  initializationU(u_host,r0,delta);

#ifdef VERIFY
  nRarray *phi_ref = new nRarray[nx];
  nRarray *u_ref = new nRarray[nx];
  memcpy(phi_ref, phi_host, vol_in_bytes);
  memcpy(u_ref, u_host, vol_in_bytes);
  reference(phi_ref, u_ref, vol, num_steps);
#endif 

  auto offload_start = std::chrono::steady_clock::now();

  

  nRarray *d_phiold = phi_host;
  nRarray *d_phinew = new nRarray[nx];
  nRarray *d_uold = u_host;
  nRarray *d_unew = new nRarray[nx];
  nRarray *d_Fx = new nRarray[nx];
  nRarray *d_Fy = new nRarray[nx];
  nRarray *d_Fz = new nRarray[nx];

  

  int t = 0;

  auto start = std::chrono::steady_clock::now();

  while (t <= num_steps) {

    calculateForce(d_phiold,d_Fx,d_Fy,d_Fz,
                   dx,dy,dz,epsilon,W0,tau0);

    allenCahn(d_phinew,d_phiold,d_uold,
              d_Fx,d_Fy,d_Fz,
              epsilon,W0,tau0,lambda,
              dt,dx,dy,dz);

    boundaryConditionsPhi(d_phinew);

    thermalEquation(d_unew,d_uold,d_phinew,d_phiold,
                    D,dt,dx,dy,dz);

    boundaryConditionsU(d_unew,delta);

    swapGrid(d_phinew, d_phiold);
    swapGrid(d_unew, d_uold);

    t++;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);

  

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
  printf("Offload time: %.3f (ms)\n", offload_time * 1e-6f);

#ifdef VERIFY
  bool ok = true;
  for (int idx = 0; idx < nx; idx++)
    for (int idy = 0; idy < ny; idy++)
      for (int idz = 0; idz < nz; idz++) {
        if (fabs(phi_ref[idx][idy][idz] - phi_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("phi: %lf %lf\n", phi_ref[idx][idy][idz], phi_host[idx][idy][idz]);
	}
        if (fabs(u_ref[idx][idy][idz] - u_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("u: %lf %lf\n", u_ref[idx][idy][idz], u_host[idx][idy][idz]);
        }
      }
  printf("%s\n", ok ? "PASS" : "FAIL");
  delete[] phi_ref;
  delete[] u_ref;
#endif

  delete[] phi_host;
  delete[] u_host;
  delete[] d_phinew;
  delete[] d_unew;
  delete[] d_Fx;
  delete[] d_Fy;
  delete[] d_Fz;

  return 0;
}