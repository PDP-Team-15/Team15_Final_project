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

__device__ double dFphi(double phi, double u, double lambda) {
    return (-phi * (1.0 - phi * phi) + lambda * u * (1.0 - phi * phi) * (1.0 - phi * phi));
}

__device__ double GradientX(double *phi, double dx, double dy, double dz, int x, int y, int z, int strideX, int strideY) {
    int idx = x * strideX + y * strideY + z;
    return (phi[idx + strideX] - phi[idx - strideX]) / (2.0 * dx);
}

__device__ double GradientY(double *phi, double dx, double dy, double dz, int x, int y, int z, int strideX, int strideY) {
    int idx = x * strideX + y * strideY + z;
    return (phi[idx + strideY] - phi[idx - strideY]) / (2.0 * dy);
}

__device__ double GradientZ(double *phi, double dx, double dy, double dz, int x, int y, int z, int strideX, int strideY) {
    int idx = x * strideX + y * strideY + z;
    return (phi[idx + 1] - phi[idx - 1]) / (2.0 * dz);
}

__device__ double Divergence(double *phix, double *phiy, double *phiz, double dx, double dy, double dz, int x, int y, int z, int strideX, int strideY) {
    return GradientX(phix, dx, dy, dz, x, y, z, strideX, strideY) +
           GradientY(phiy, dx, dy, dz, x, y, z, strideX, strideY) +
           GradientZ(phiz, dx, dy, dz, x, y, z, strideX, strideY);
}

__device__ double Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z, int strideX, int strideY) {
    int idx = x * strideX + y * strideY + z;
    double phixx = (phi[idx + strideX] + phi[idx - strideX] - 2.0 * phi[idx]) / SQ(dx);
    double phiyy = (phi[idx + strideY] + phi[idx - strideY] - 2.0 * phi[idx]) / SQ(dy);
    double phizz = (phi[idx + 1] + phi[idx - 1] - 2.0 * phi[idx]) / SQ(dz);
    return phixx + phiyy + phizz;
}

__device__ double An(double phix, double phiy, double phiz, double epsilon) {
    if (phix != 0.0 || phiy != 0.0 || phiz != 0.0) {
        double numerator = 4.0 * epsilon / (1.0 - 3.0 * epsilon);
        double denominator = (SQ(phix) + SQ(phiy) + SQ(phiz)) * (SQ(phix) + SQ(phiy) + SQ(phiz));
        double term = numerator * (SQ(phix) * SQ(phix) + SQ(phiy) * SQ(phiy) + SQ(phiz) * SQ(phiz)) / denominator;
        return (1.0 - 3.0 * epsilon) * (1.0 + term);
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
        double numerator = l * l * l * (SQ(m) + SQ(n)) - l * (SQ(m) * SQ(m) + SQ(n) * SQ(n));
        double denominator = (SQ(l) + SQ(m) + SQ(n)) * (SQ(l) + SQ(m) + SQ(n));
        return numerator / denominator;
    } else {
        return 0.0;
    }
}

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, double epsilon, double W0, double tau0, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x < DATAXSIZE - 1 && y < DATAYSIZE - 1 && z < DATAZSIZE - 1 && x > 0 && y > 0 && z > 0) {
            double phix = GradientX(phi, dx, dy, dz, x, y, z, strideX, strideY);
            double phiy = GradientY(phi, dx, dy, dz, x, y, z, strideX, strideY);
            double phiz = GradientZ(phi, dx, dy, dz, x, y, z, strideX, strideY);
            double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
            double c = 16.0 * W0 * epsilon;
            double w = Wn(phix, phiy, phiz, epsilon, W0);
            double w2 = SQ(w);

            int idx = x * strideX + y * strideY + z;
            Fx[idx] = w2 * phix + sqGphi * w * c * dFunc(phix, phiy, phiz);
            Fy[idx] = w2 * phiy + sqGphi * w * c * dFunc(phiy, phiz, phix);
            Fz[idx] = w2 * phiz + sqGphi * w * c * dFunc(phiz, phix, phiy);
        } else {
            int idx = x * strideX + y * strideY + z;
            Fx[idx] = 0.0;
            Fy[idx] = 0.0;
            Fz[idx] = 0.0;
        }
    }
}

__global__ void allenCahn(double *phinew, double *phiold, double *uold, double *Fx, double *Fy, double *Fz, double epsilon, double W0, double tau0, double lambda, double dt, double dx, double dy, double dz, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE - 1 && y < DATAYSIZE - 1 && z < DATAZSIZE - 1) {
        double phix = GradientX(phiold, dx, dy, dz, x, y, z, strideX, strideY);
        double phiy = GradientY(phiold, dx, dy, dz, x, y, z, strideX, strideY);
        double phiz = GradientZ(phiold, dx, dy, dz, x, y, z, strideX, strideY);

        int idx = x * strideX + y * strideY + z;
        double divergence = Divergence(Fx, Fy, Fz, dx, dy, dz, x, y, z, strideX, strideY);
        double dfphi = dFphi(phiold[idx], uold[idx], lambda);
        phinew[idx] = phiold[idx] + (dt / taun(phix, phiy, phiz, epsilon, tau0)) * (divergence - dfphi);
    }
}

__global__ void boundaryConditionsPhi(double *phinew, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x == 0 || x == DATAXSIZE - 1 || y == 0 || y == DATAYSIZE - 1 || z == 0 || z == DATAZSIZE - 1) {
            int idx = x * strideX + y * strideY + z;
            phinew[idx] = -1.0;
        }
    }
}

__global__ void thermalEquation(double *unew, double *uold, double *phinew, double *phiold, double D, double dt, double dx, double dy, double dz, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE - 1 && y < DATAYSIZE - 1 && z < DATAZSIZE - 1) {
        double laplacian = Laplacian(uold, dx, dy, dz, x, y, z, strideX, strideY);
        int idx = x * strideX + y * strideY + z;
        unew[idx] = uold[idx] + 0.5 * (phinew[idx] - phiold[idx]) + dt * D * laplacian;
    }
}

__global__ void boundaryConditionsU(double *unew, double delta, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x == 0 || x == DATAXSIZE - 1 || y == 0 || y == DATAYSIZE - 1 || z == 0 || z == DATAZSIZE - 1) {
            int idx = x * strideX + y * strideY + z;
            unew[idx] = -delta;
        }
    }
}

__global__ void swapGrid(double *cnew, double *cold, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        int idx = x * strideX + y * strideY + z;
        double tmp = cnew[idx];
        cnew[idx] = cold[idx];
        cold[idx] = tmp;
    }
}

__global__ void initializationPhi(double *phi, double r0, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        double xc = x - 0.5 * DATAXSIZE;
        double yc = y - 0.5 * DATAYSIZE;
        double zc = z - 0.5 * DATAZSIZE;
        double r = sqrt(SQ(xc) + SQ(yc) + SQ(zc));
        int idx = x * strideX + y * strideY + z;
        phi[idx] = (r < r0) ? 1.0 : -1.0;
    }
}

__global__ void initializationU(double *u, double r0, double delta, int strideX, int strideY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        double xc = x - 0.5 * DATAXSIZE;
        double yc = y - 0.5 * DATAYSIZE;
        double zc = z - 0.5 * DATAZSIZE;
        double r = sqrt(SQ(xc) + SQ(yc) + SQ(zc));
        int idx = x * strideX + y * strideY + z;
        u[idx] = (r < r0) ? 0.0 : -delta * (1.0 - exp(-(r - r0)));
    }
}

int main(int argc, char *argv[])
{
  const int num_steps = atoi(argv[1]);  //6000;
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

  // overall data set sizes
  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_in_bytes = sizeof(double) * vol;

  // pointers for data set storage via malloc
  nRarray *phi_host;
  nRarray *d_phiold;
  nRarray *u_host;
  nRarray *d_phinew;
  nRarray *d_uold;
  nRarray *d_unew;
  nRarray *d_Fx;
  nRarray *d_Fy;
  nRarray *d_Fz;

  phi_host = (nRarray *)malloc(vol_in_bytes);
  u_host = (nRarray *)malloc(vol_in_bytes);

  initializationPhi(phi_host,r0);
  initializationU(u_host,r0,delta);

#ifdef VERIFY
  nRarray *phi_ref = (nRarray *)malloc(vol_in_bytes);
  nRarray *u_ref = (nRarray *)malloc(vol_in_bytes);
  memcpy(phi_ref, phi_host, vol_in_bytes);
  memcpy(u_ref, u_host, vol_in_bytes);
  reference(phi_ref, u_ref, vol, num_steps);
#endif

  auto offload_start = std::chrono::steady_clock::now();

  // define the chunk sizes that each threadblock will work on
  dim3 grid ((DATAZSIZE+7)/8, (DATAYSIZE+7)/8, (DATAXSIZE+3)/4);
  dim3 block (8, 8, 4);

  // allocate GPU device buffers
  cudaMalloc((void **) &d_phiold, vol_in_bytes);
  cudaMalloc((void **) &d_phinew, vol_in_bytes);
  cudaMalloc((void **) &d_uold, vol_in_bytes);
  cudaMalloc((void **) &d_unew, vol_in_bytes);
  cudaMalloc((void **) &d_Fx, vol_in_bytes);
  cudaMalloc((void **) &d_Fy, vol_in_bytes);
  cudaMalloc((void **) &d_Fz, vol_in_bytes);

  cudaMemcpy(d_phiold, phi_host, vol_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uold, u_host, vol_in_bytes, cudaMemcpyHostToDevice);

  int t = 0;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (t <= num_steps) {

    calculateForce<<<grid, block>>>(d_phiold,d_Fx,d_Fy,d_Fz,
                                    dx,dy,dz,epsilon,W0,tau0);

    allenCahn<<<grid, block>>>(d_phinew,d_phiold,d_uold,
                               d_Fx,d_Fy,d_Fz,
                               epsilon,W0,tau0,lambda,
                               dt,dx,dy,dz);

    boundaryConditionsPhi<<<grid, block>>>(d_phinew);

    thermalEquation<<<grid, block>>>(d_unew,d_uold,d_phinew,d_phiold,
                                     D,dt,dx,dy,dz);

    boundaryConditionsU<<<grid, block>>>(d_unew,delta);

    swapGrid<<<grid, block>>>(d_phinew, d_phiold);

    swapGrid<<<grid, block>>>(d_unew, d_uold);

    t++;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);

  cudaMemcpy(phi_host, d_phiold, vol_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(u_host, d_uold, vol_in_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_phiold);
  cudaFree(d_phinew);
  cudaFree(d_uold);
  cudaFree(d_unew);
  cudaFree(d_Fx);
  cudaFree(d_Fy);
  cudaFree(d_Fz);

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
  free(phi_ref);
  free(u_ref);
#endif

  free(phi_host);
  free(u_host);
  return 0;
}