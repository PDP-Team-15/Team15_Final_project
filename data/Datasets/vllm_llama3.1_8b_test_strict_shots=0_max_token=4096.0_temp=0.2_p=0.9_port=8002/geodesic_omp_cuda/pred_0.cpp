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

float  distance_host ( int i, float latitude_1, float longitude_1,
                       float latitude_2, float longitude_2 )
{
  // ... (no changes)
}

__global__ void distance_device(const float4* VA, float* VC, const int N, const int iteration) {

  __shared__ float GDC_DEG_TO_RAD;
  __shared__ float GDC_FLATTENING;
  __shared__ float GDC_ECCENTRICITY;
  __shared__ float GDC_ELLIPSOIDAL;
  __shared__ float GDC_SEMI_MINOR;
  __shared__ float EPS;

  __shared__ float TU1;
  __shared__ float TU2;
  __shared__ float CU1;
  __shared__ float CU2;
  __shared__ float BAZ;
  __shared__ float FAZ;
  __shared__ float C;
  __shared__ float C2A;
  __shared__ float SX;
  __shared__ float CX;
  __shared__ float SY;
  __shared__ float CY;
  __shared__ float CZ;
  __shared__ float D;
  __shared__ float E;
  __shared__ float SA;
  __shared__ float SU1;
  __shared__ float X;
  __shared__ float Y;

  __shared__ float rad_latitude_1;
  __shared__ float rad_longitude_1;
  __shared__ float rad_latitude_2;
  __shared__ float rad_longitude_2;

  const int wiID = blockIdx.x * blockDim.x + threadIdx.x;

  if (wiID < N) {
    GDC_DEG_TO_RAD = 3.141592654 / 180.0;
    GDC_FLATTENING = 1.0 - ( 6356752.31424518 / 6378137.0 );
    GDC_ECCENTRICITY = ( 6356752.31424518 / 6378137.0 );
    GDC_ELLIPSOIDAL =  1.0 / ( 6356752.31414 / 6378137.0 ) / ( 6356752.31414 / 6378137.0 ) - 1.0;
    GDC_SEMI_MINOR = 6356752.31424518;
    EPS = 0.5e-5;

    rad_latitude_1 = VA[wiID].x * GDC_DEG_TO_RAD;
    rad_longitude_1 = VA[wiID].y * GDC_DEG_TO_RAD;
    rad_latitude_2 = VA[wiID].z * GDC_DEG_TO_RAD;
    rad_longitude_2 = VA[wiID].w * GDC_DEG_TO_RAD;

    TU1 = GDC_ECCENTRICITY * sinf(rad_latitude_1) / cosf(rad_latitude_1);
    TU2 = GDC_ECCENTRICITY * sinf(rad_latitude_2) / cosf(rad_latitude_2);

    CU1 = 1.0f / sqrtf(TU1 * TU1 + 1.0f);
    SU1 = CU1 * TU1;
    CU2 = 1.0f / sqrtf(TU2 * TU2 + 1.0f);
    __shared__ float dist;
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
    dist = ((SY * SY * 4.0f - 3.0f) * dist * CZ * D / 6.0f - X) * D / 4.0f + CZ) * SY * D + Y) * C * GDC_SEMI_MINOR;
    VC[wiID] = dist;
  }
}

void verify(int size, const float *output, const float *expected_output) {
  // ... (no changes)
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

  for (int i = 0; i < N; i++) {
    float lat1 = input[i].x;
    float lon1 = input[i].y;
    float lat2 = input[i].z;
    float lon2 = input[i].w;
    expected_output[i] = distance_host(i, lat1, lon1, lat2, lon2);
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  distance_device<<<numBlocks, blockSize>>>(input, output, N, iteration);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Average kernel execution time %f (s)\n", milliseconds / 1000.0f / iteration);

  verify(N, output, expected_output);

  free(input);
  free(output);
  free(expected_output);
  return 0;
}
```