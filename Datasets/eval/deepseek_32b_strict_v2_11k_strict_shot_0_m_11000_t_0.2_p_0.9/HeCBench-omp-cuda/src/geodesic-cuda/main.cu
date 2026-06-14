#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

float distance_host(int i, float latitude_1, float longitude_1,
                    float latitude_2, float longitude_2)
{
  // ... (same as original code)
}

__device__ void compute_distance(float rad_latitude_1, float rad_longitude_1,
                                float rad_latitude_2, float rad_longitude_2,
                                float* dist)
{
  const float GDC_DEG_TO_RAD = 3.141592654f / 180.0f;
  const float GDC_FLATTENING = 1.0f - (6356752.31424518f / 6378137.0f);
  const float GDC_ECCENTRICITY = 6356752.31424518f / 6378137.0f;
  const float GDC_ELLIPSOIDAL = 1.0f / (6356752.31414f / 6378137.0f) / (6356752.31414f / 6378137.0f) - 1.0f;
  const float GDC_SEMI_MINOR = 6356752.31424518f;
  const float EPS = 0.5e-5f;

  float TU1, TU2, CU1, SU1, CU2, BAZ, FAZ, X, SX, CX, TU1_tmp, TU2_tmp, SY, CY, Y, SA, C2A, CZ, E, C, D;

  TU1 = GDC_ECCENTRICITY * sinf(rad_latitude_1) / cosf(rad_latitude_1);
  TU2 = GDC_ECCENTRICITY * sinf(rad_latitude_2) / cosf(rad_latitude_2);

  CU1 = 1.0f / sqrtf(TU1 * TU1 + 1.0f);
  SU1 = CU1 * TU1;
  CU2 = 1.0f / sqrtf(TU2 * TU2 + 1.0f);

  *dist = CU1 * CU2;
  BAZ = *dist * TU2;
  FAZ = BAZ * TU1;
  X = rad_longitude_2 - rad_longitude_1;

  do {
    SX = sinf(X);
    CX = cosf(X);
    TU1_tmp = CU2 * SX;
    TU2_tmp = BAZ - SU1 * CU2 * CX;
    SY = sqrtf(TU1_tmp * TU1_tmp + TU2_tmp * TU2_tmp);
    CY = *dist * CX + FAZ;
    Y = atan2f(SY, CY);
    SA = *dist * SX / SY;
    C2A = -SA * SA + 1.0f;
    CZ = FAZ + FAZ;
    if (C2A > 0.0f) CZ = -CZ / C2A + CY;
    E = CZ * CZ * 2.0f - 1.0f;
    C = ((-3.0f * C2A + 4.0f) * GDC_FLATTENING + 4.0f) * C2A * GDC_FLATTENING / 16.0f;
    D = X;
    X = ((E * CY * C + CZ) * SY * C + Y) * SA;
    X = (1.0f - C) * X * GDC_FLATTENING + rad_longitude_2 - rad_longitude_1;
  } while (fabsf(D - X) > EPS);

  X = sqrtf(GDC_ELLIPSOIDAL * C2A + 1.0f) + 1.0f;
  X = (X - 2.0f) / X;
  C = 1.0f - X;
  C = (X * X / 4.0f + 1.0f) / C;
  D = (0.375f * X * X - 1.0f) * X;
  X = E * CY;
  *dist = 1.0f - E - E;
  *dist = (((SY * SY * 4.0f - 3.0f) * *dist * CZ * D / 6.0f - X) * D / 4.0f + CZ) * SY * D + Y;
  *dist = *dist * C * GDC_SEMI_MINOR;
}

__global__ void distance_kernel(const float4* VA, float* VC, const size_t N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float rad_latitude_1 = VA[idx].x * (3.141592654f / 180.0f);
    float rad_longitude_1 = VA[idx].y * (3.141592654f / 180.0f);
    float rad_latitude_2 = VA[idx].z * (3.141592654f / 180.0f);
    float rad_longitude_2 = VA[idx].w * (3.141592654f / 180.0f);
    
    float dist;
    compute_distance(rad_latitude_1, rad_longitude_1, rad_latitude_2, rad_longitude_2, &dist);
    VC[idx] = dist;
  }
}

void distance_device(const float4* VA, float* VC, const size_t N, const int iteration)
{
  float4* d_VA;
  float* d_VC;
  cudaMalloc((void**)&d_VA, N * sizeof(float4));
  cudaMalloc((void**)&d_VC, N * sizeof(float));
  
  cudaMemcpy(d_VA, VA, N * sizeof(float4), cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(256);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  for (int n = 0; n < iteration; n++) {
    distance_kernel<<<numBlocks, threadsPerBlock>>>(d_VA, d_VC, N);
    cudaDeviceSynchronize();
  }
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  
  cudaMemcpy(VC, d_VC, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  printf("Average kernel execution time %f (s)\n", time / iteration / 1000.0f);
  
  cudaFree(d_VA);
  cudaFree(d_VC);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void verify(int size, const float* output, const float* expected_output)
{
  float error_rate = 0;
  for (int i = 0; i < size; i++) {
    if (fabs(output[i] - expected_output[i]) > error_rate) {
      error_rate = fabs(output[i] - expected_output[i]);
    }
  }
  printf("The maximum error in distance for single precision is %f\n", error_rate);
}

int main(int argc, char** argv)
{
  // ... (same as original code)
}