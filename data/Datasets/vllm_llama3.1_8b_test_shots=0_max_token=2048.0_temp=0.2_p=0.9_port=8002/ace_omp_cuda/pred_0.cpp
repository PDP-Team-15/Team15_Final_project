Here is the CUDA version of the provided code:

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

__global__ void dFphiKernel(double *phi, double *u, double *lambda, double *dFphi, int nx, int ny, int nz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= 0 && idx < nx && idy >= 0 && idy < ny && idz >= 0 && idz < nz) {
    int i = idx;
    int j = idy;
    int k = idz;

    double phi_val = phi[i * ny * nz + j * nz + k];
    double u_val = u[i * ny * nz + j * nz + k];
    double lambda_val = lambda[0];

    dFphi[i * ny * nz + j * nz + k] = (-phi_val * (1.0 - phi_val * phi_val) + lambda_val * u_val * (1.0 - phi_val * phi_val) * (1.0 - phi_val * phi_val));
  }
}

__global__ void GradientXKernel(double *phi, double *dx, double *dy, double *dz, double *GradientX, int nx, int ny, int nz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= 0 && idx < nx && idy >= 0 && idy < ny && idz >= 0 && idz < nz) {
    int i = idx;
    int j = idy;
    int k = idz;

    double phi_val = phi[i * ny * nz + j * nz + k];
    double dx_val = dx[0];
    double dy_val = dy[0];
    double dz_val = dz[0];

    GradientX[i * ny * nz + j * nz + k] = (phi[(i + 1) * ny * nz + j * nz + k] - phi[(i - 1) * ny * nz + j * nz + k]) / (2.0 * dx_val);
  }
}

__global__ void GradientYKernel(double *phi, double *dx, double *dy, double *dz, double *GradientY, int nx, int ny, int nz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= 0 && idx < nx && idy >= 0 && idy < ny && idz >= 0 && idz < nz) {
    int i = idx;
    int j = idy;
    int k = idz;

    double phi_val = phi[i * ny * nz + j * nz + k];
    double dx_val = dx[0];
    double dy_val = dy[0];
    double dz_val = dz[0];

    GradientY[i * ny * nz + j * nz + k] = (phi[i * ny * nz + (j + 1) * nz + k] - phi[i * ny * nz + (j - 1) * nz + k]) / (2.0 * dy_val);
  }
}

__global__ void GradientZKernel(double *phi, double *dx, double *dy, double *dz, double *GradientZ, int nx, int ny, int nz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= 0 && idx < nx && idy >= 0 && idy < ny && idz >= 0 && idz < nz) {
    int i = idx;
    int j = idy;
    int k = idz;

    double phi_val = phi[i * ny * nz + j * nz + k];
    double dx_val = dx[0];
    double dy_val = dy[0];
    double dz_val = dz[0];

    GradientZ[i * ny * nz + j * nz + k] = (phi[i * ny * nz + j * nz + (k + 1)] - phi[i * ny * nz + j * nz + (k - 1)]) / (2.0 * dz_val);
  }
}

__global__ void DivergenceKernel(double *Fx, double *Fy, double *Fz, double *dx, double *dy, double *dz, double *Divergence, int nx, int ny, int nz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= 0 && idx < nx && idy >= 0 && idy < ny && idz >= 0 && idz < nz) {
    int i = idx;
    int j = idy;
    int k = idz;

    double Fx_val = Fx[i * ny * nz + j * nz + k];
    double Fy_val = Fy[i * ny * nz + j * nz + k];
    double Fz_val = Fz[i * ny * nz + j * nz + k];
    double dx_val = dx[0];
    double dy_val = dy[0];
    double dz_val = dz[0];

    Divergence[i * ny * nz + j * nz + k] = GradientXKernel<<<dim3(nx / 16, ny / 16, nz / 16), dim3(16, 16, 16)>>>(Fx, dx, dy, dz, Fx, nx, ny, nz).x;
    Divergence[i * ny * nz + j * nz + k] += GradientYKernel<<<dim3(nx / 16, ny / 16, nz / 16), dim3(16, 16, 16)>>>(Fy, dx, dy, dz, Fy, nx, ny, nz).x;
    Divergence[i * ny * nz + j * nz + k] += GradientZKernel<<<dim3(nx / 16, ny / 16, nz / 16), dim3(16, 16, 16)>>>(Fz, dx, dy, dz, Fz, nx, ny, nz).x;
  }
}

__global__ void LaplacianKernel(double *phi, double *dx, double *dy, double *dz, double *Laplacian, int nx, int ny, int nz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= 0 && idx < nx && idy >= 0 && idy < ny && idz >= 0 && idz < nz) {
    int i = idx;
    int j = idy;
    int k = idz;

    double phi_val = phi[i * ny * nz + j * nz + k];
    double dx_val = dx[0];
    double dy_val = dy[0];
    double dz_val = dz[0];

    double phixx = (phi[(i + 1) * ny * nz + j * nz + k] + phi[(i - 1) * ny * nz + j * nz + k] - 2.0 * phi_val) / SQ(dx_val);
    double phiyy = (phi[i * ny * nz + (j + 1) * nz + k] + phi[i * ny * nz + (j - 1) * nz + k] - 2.0 * phi_val) / SQ(dy_val);
    double phizz = (phi[i * ny * nz + j * nz + (k + 1)] + phi[i * ny * nz + j * nz + (k - 1)] - 2.0 * phi_val) / SQ(dz_val);

    Laplacian[i * ny * nz + j * nz + k] = phixx + phiyy + phizz;
  }
}

__global__ void AnKernel(double *phix, double *phiy, double *phiz, double *epsilon, double *An, int nx, int ny, int nz) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= 0 && idx < nx && idy >= 0 && idy < ny && idz >= 0 && idz < nz) {
    int i = idx;
    int j = idy;
    int k = idz;

    double phix_val = phix[i * ny * nz + j * nz + k];
    double phiy_val = ph