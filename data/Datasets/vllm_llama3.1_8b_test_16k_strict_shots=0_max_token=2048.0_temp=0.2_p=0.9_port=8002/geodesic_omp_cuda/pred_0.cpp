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
  __shared__ float X;
  __shared__ float SX;
  __shared__ float CX;
  __shared__ float SY;
  __shared__ float CY;
  __shared__ float Y;
  __shared__ float SA;
  __shared__ float C2A;
  __shared__ float CZ;
  __shared__ float D;
  __shared__ float E;

  __shared__ float rad_latitude_1;
  __shared__ float rad_longitude_1;
  __shared__ float rad_latitude_2;
  __shared__ float rad_longitude_2;

  __shared__ float dist;

  __shared__ float4 VA_thread;
  __shared__ float VC_thread;

  __shared__ int wiID;

  __shared__ int iteration_thread;

  __shared__ int N_thread;

  __shared__ int index;

  __shared__ int num_ref_cities;

  __shared__ int num_cities;

  __shared__ int index_map[] ={436483, 1952407, 627919, 377884, 442703, 1863423};

  __shared__ int city;

  __shared__ float lat;

  __shared__ float lon;

  __shared__ const char* filename;

  __shared__ FILE* fp;

  __shared__ float4* input;

  __shared__ float*  output;

  __shared__ float*  expected_output;

  __shared__ int size;

  __shared__ float error_rate;

  __shared__ float time;

  __shared__ int thread_id;

  __shared__ int num_threads;

  __shared__ int block_id;

  __shared__ int grid_size;

  __shared__ int block_size;

  __shared__ int index_block;

  __shared__ int index_thread;

  __shared__ int index_grid;

  __shared__ int index_block_thread;

  __shared__ int index_grid_thread;

  __shared__ int index_block_grid;

  __shared__ int index_thread_block;

  __shared__ int index_grid_block;

  __shared__ int index_thread_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int index_thread_block_grid;

  __shared__ int index_block_thread_grid;

  __shared__ int index_grid_thread_block;

  __shared__ int index_block_grid_thread;

  __shared__ int index_thread_grid_block;

  __shared__ int index_grid_block_thread;

  __shared__ int