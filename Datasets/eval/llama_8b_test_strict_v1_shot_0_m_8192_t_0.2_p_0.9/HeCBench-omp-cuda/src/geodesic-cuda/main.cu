#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

__device__ float distance_device(const float4* VA, const int wiID, const float GDC_DEG_TO_RAD, 
                                 const float GDC_FLATTENING, const float GDC_ECCENTRICITY, 
                                 const float GDC_ELLIPSOIDAL, const float GDC_SEMI_MINOR, const float EPS) {

  float  dist;
  float  rad_latitude_1;
  float  rad_longitude_1;
  float  rad_latitude_2;
  float  rad_longitude_2;

  float  BAZ, C, C2A, CU1, CU2, CX, CY, CZ, D, E, FAZ, SA, SU1, SX, SY, TU1, TU2, X, Y;

  rad_longitude_1 = VA[wiID].y * GDC_DEG_TO_RAD;
  rad_latitude_1 = VA[wiID].x * GDC_DEG_TO_RAD;
  rad_longitude_2 = VA[wiID].w * GDC_DEG_TO_RAD;
  rad_latitude_2 = VA[wiID].z * GDC_DEG_TO_RAD;

  TU1 = GDC_ECCENTRICITY * sinf(rad_latitude_1) / cosf(rad_latitude_1);
  TU2 = GDC_ECCENTRICITY * sinf(rad_latitude_2) / cosf(rad_latitude_2);

  CU1 = 1.0f / sqrtf(TU1 * TU1 + 1.0f);
  SU1 = CU1 * TU1;
  CU2 = 1.0f / sqrtf(TU2 * TU2 + 1.0f);
  dist = CU1 * CU2;
  BAZ = dist * TU2;
  FAZ = BAZ * TU1;
  X = rad_longitude_2 - rad_longitude_1;

  do {
    SX = sinf(X);
    CX = cosf(X);
    TU1 = CU2 * SX;
    TU2 = BAZ - SU1 * CU2 * CX;
    SY = sqrtf(TU1 * TU1 + TU2 * TU2);
    CY = dist * CX + FAZ;
    Y = atan2f(SY, CY);
    SA = dist * SX / SY;
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
  dist = 1.0f - E - E;
  dist = (((((SY * SY * 4.0f - 3.0f) * dist * CZ * D / 6.0f - X) * D / 4.0f + CZ) * SY * D + Y) * C * GDC_SEMI_MINOR;
  return dist;
}

__global__ void distance_kernel(float4* VA, float* VC, const int N, const int iteration, 
                                const float GDC_DEG_TO_RAD, const float GDC_FLATTENING, 
                                const float GDC_ECCENTRICITY, const float GDC_ELLIPSOIDAL, 
                                const float GDC_SEMI_MINOR, const float EPS) {

  const int wiID = blockIdx.x * blockDim.x + threadIdx.x;
  if (wiID < N) {
    float dist = distance_device(VA, wiID, GDC_DEG_TO_RAD, GDC_FLATTENING, GDC_ECCENTRICITY, 
                                 GDC_ELLIPSOIDAL, GDC_SEMI_MINOR, EPS);
    VC[wiID] = dist;
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage %s <repeat>\n", argv[0]);
    return 1;
  }
  int iteration = atoi(argv[1]);

  int num_cities = 2097152;
  int num_ref_cities = 6;
  int index_map[] = {436483, 1952407, 627919, 377884, 442703, 1863423};
  int N = num_cities * num_ref_cities;
  int city = 0;
  float lat, lon;

  const char* filename = "locations.txt";
  printf("Reading city locations from file %s...\n", filename);
  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    perror("Error opening the file");
    exit(-1);
  }

  float4* input;
  cudaMallocHost((void**)&input, N * sizeof(float4));
  float* output;
  cudaMallocHost((void**)&output, N * sizeof(float));
  float* expected_output;
  cudaMallocHost((void**)&expected_output, N * sizeof(float));

  while (fscanf(fp, "%f %f\n", &lat, &lon) != EOF) {
    input[city].x = lat;
    input[city].y = lon;
    city++;
    if (city == num_cities) break;
  }
  fclose(fp);

  for (int c = 1; c < num_ref_cities; c++) {
    std::copy(input, input + num_cities, input + c * num_cities);
  }

  for (int c = 0; c < num_ref_cities; c++) {
    int index = index_map[c] - 1;
    for (int j = c * num_cities; j < (c + 1) * num_cities; ++j) {
      input[j].z = input[index].x;
      input[j].w = input[index].y;
    }
  }

  for (int i = 0; i < N; i++) {
    float lat1 = input[i].x;
    float lon1 = input[i].y;
    float lat2 = input[i].z;
    float lon2 = input[i].w;
    expected_output[i] = distance_host(i, lat1, lon1, lat2, lon2);
  }

  float GDC_DEG_TO_RAD = 3.141592654 / 180.0;
  float GDC_FLATTENING = 1.0 - (6356752.31424518 / 6378137.0);
  float GDC_ECCENTRICITY = (6356752.31424518 / 6378137.0);
  float GDC_ELLIPSOIDAL = 1.0 / (6356752.31414 / 6378137.0) / (6356752.31414 / 6378137.0) - 1.0;
  float GDC_SEMI_MINOR = 6356752.31424518;
  float EPS = 0.5e-5;

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  distance_kernel<<<numBlocks, blockSize>>>(input, output, N, iteration, GDC_DEG_TO_RAD, 
                                             GDC_FLATTENING, GDC_ECCENTRICITY, GDC_ELLIPSOIDAL, 
                                             GDC_SEMI_MINOR, EPS);
  cudaDeviceSynchronize();

  float error_rate = 0;
  for (int i = 0; i < N; i++) {
    if (fabs(output[i] - expected_output[i]) > error_rate) {
      error_rate = fabs(output[i] - expected_output[i]);
    }
  }
  printf("The maximum error in distance for single precision is %f\n", error_rate);

  cudaFreeHost(input);
  cudaFreeHost(output);
  cudaFreeHost(expected_output);
  return 0;
}