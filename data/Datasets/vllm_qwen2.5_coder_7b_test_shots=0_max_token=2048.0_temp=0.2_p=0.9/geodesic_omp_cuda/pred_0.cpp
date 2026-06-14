Here's the CUDA translation of your OpenMP code:

```cpp
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

__device__ float distance_device_kernel(const float4& va) {
  const float GDC_DEG_TO_RAD = 3.141592654 / 180.0;  
  const float GDC_FLATTENING = 1.0 - (6356752.31424518 / 6378137.0); 
  const float GDC_ECCENTRICITY = (6356752.31424518 / 6378137.0); 
  const float GDC_ELLIPSOIDAL = 1.0 / (6356752.31414 / 6378137.0) / (6356752.31414 / 6378137.0) - 1.0;
  const float GDC_SEMI_MINOR = 6356752.31424518f;
  const float EPS = 0.5e-5f;

  float rad_longitude_1 = va.x * GDC_DEG_TO_RAD;
  float rad_latitude_1 = va.y * GDC_DEG_TO_RAD;
  float rad_longitude_2 = va.z * GDC_DEG_TO_RAD;
  float rad_latitude_2 = va.w * GDC_DEG_TO_RAD;

  float TU1 = GDC_ECCENTRICITY * sinf(rad_latitude_1) / cosf(rad_latitude_1);
  float TU2 = GDC_ECCENTRICITY * sinf(rad_latitude_2) / cosf(rad_latitude_2);

  float CU1 = 1.0f / sqrtf(TU1 * TU1 + 1.0f);
  float SU1 = CU1 * TU1;
  float CU2 = 1.0f / sqrtf(TU2 * TU2 + 1.0f);
  float dist = CU1 * CU2;
  float BAZ = dist * TU2;
  float FAZ = BAZ * TU1;
  float X = rad_longitude_2 - rad_longitude_1;

  float SX, CX, TU1_new, TU2_new, SY, CY, Y, SA, C2A, CZ, E, C, D;

  do {
    SX = sinf(X);
    CX = cosf(X);
    TU1_new = CU2 * SX;
    TU2_new = BAZ - SU1 * CU2 * CX;
    SY = sqrtf(TU1_new * TU1_new + TU2_new * TU2_new);
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
  dist = (((SY * SY * 4.0f - 3.0f) * dist * CZ * D / 6.0f - X) * D / 4.0f + CZ) * SY * D + Y) * C * GDC_SEMI_MINOR;
  return dist;
}

__global__ void distance_device_kernel_gpu(const float4* VA, float* VC, const size_t N) {
  int wiID = blockIdx.x * blockDim.x + threadIdx.x;
  if (wiID < N) {
    VC[wiID] = distance_device_kernel(VA[wiID]);
  }
}

void distance_device(const float4* VA, float* VC, const size_t N, const int iteration) {
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < iteration; n++) {
    distance_device_kernel_gpu<<<(N + 255) / 256, 256>>>(VA, VC, N);
  }

  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);
}

void verify(int size, const float *output, const float *expected_output) {
  float error_rate = 0;
  for (int i = 0; i < size; i++) {
    if (fabs(output[i] - expected_output[i]) > error_rate) {
      error_rate = fabs(output[i] - expected_output[i]);
    }
  }
  printf("The maximum error in distance for single precision is %f\n", error_rate); 
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage %s <repeat>\n", argv[0]);
    return 1;
  }
  int iteration = atoi(argv[1]);

  int num_cities = 2097152; 
  int num_ref_cities = 6; 
  int index_map[] ={436483, 1952407, 627919, 377884, 442703, 1863423};
  int N = num_cities * num_ref_cities;
  int city = 0;
  float lat, lon;

  const char* filename = "locations.txt";
  printf("Reading city locations from file %s...\n", filename);
  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    perror ("Error opening the file");
    exit(-1);
  }

  float4* input  = (float4*) aligned_alloc(4096, N*sizeof(float4));
  float*  output = (float*) aligned_alloc(4096, N*sizeof(float));
  float*  expected_output = (float*) malloc(N*sizeof(float));

  while (fscanf(fp, "%f %f\n", &lat, &lon) != EOF) { 
    input[city].x = lat;
    input[city].y = lon;
    city++;
    if (city == num_cities) break;  
  }
  fclose(fp);

  for (int c = 1;  c < num_ref_cities; c++) {
    std::copy(input, input+num_cities, input+c*num_cities);
  }

  for (int c = 0;  c < num_ref_cities; c++) {
    int index = index_map[c] - 1;
    for(int j = c*num_cities; j < (c+1)*num_cities; ++j) {
      input[j].z = input[index].x;
      input[j].w = input[index].y;
    }
  }

  for (int i = 0; i < N; i++)
  {
    float lat1 = input[i].x;
    float lon1 = input[i].y;
    float lat2 = input[i].z;
    float lon2 = input[i].w;
    expected_output[i] = distance_host(i, lat1, lon1, lat2, lon2);
  }

  distance_device(input, output, N, iteration);

  verify(N, output, expected_output);

  free(input);
  free(output);
  free(expected_output);
  return 0;
}
```

This CUDA version includes a device function `distance_device_kernel` that computes the distance for a single pair of coordinates. The `distance_device_kernel_gpu` kernel launches this function for each element in the input array. The rest of the code remains largely unchanged, ensuring that the main functionality and timing remain consistent with the original OpenMP implementation.