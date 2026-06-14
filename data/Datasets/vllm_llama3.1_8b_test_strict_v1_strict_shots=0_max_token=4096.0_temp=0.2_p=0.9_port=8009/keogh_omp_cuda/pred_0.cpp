```cuda
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include "reference.h"

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return -1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  float *subject;
  float *lower_bound;
  float *upper_bound;
  float *lb;
  float *lb_h;
  float *avgs;
  float *stds;

  CUDA_CHECK(cudaMalloc((void**)&subject, sizeof(float)*N));
  CUDA_CHECK(cudaMalloc((void**)&lower_bound, sizeof(float)*N));
  CUDA_CHECK(cudaMalloc((void**)&upper_bound, sizeof(float)*N));
  CUDA_CHECK(cudaMalloc((void**)&lb, sizeof(float)*(N-M+1)));
  CUDA_CHECK(cudaMalloc((void**)&lb_h, sizeof(float)*(N-M+1)));
  CUDA_CHECK(cudaMalloc((void**)&avgs, sizeof(float)*(N-M+1)));
  CUDA_CHECK(cudaMalloc((void**)&stds, sizeof(float)*(N-M+1)));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper_bound[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower_bound[i] = (float)rand() / (float)RAND_MAX;

  const int blocks = 256;
  const int grids = (N-M+1 + blocks - 1) / blocks;

  float *d_subject, *d_avgs, *d_stds, *d_lower_bound, *d_upper_bound;
  CUDA_CHECK(cudaMalloc((void**)&d_subject, sizeof(float)*N));
  CUDA_CHECK(cudaMalloc((void**)&d_avgs, sizeof(float)*(N-M+1)));
  CUDA_CHECK(cudaMalloc((void**)&d_stds, sizeof(float)*(N-M+1)));
  CUDA_CHECK(cudaMalloc((void**)&d_lower_bound, sizeof(float)*N));
  CUDA_CHECK(cudaMalloc((void**)&d_upper_bound, sizeof(float)*N));

  CUDA_CHECK(cudaMemcpy(d_subject, subject, sizeof(float)*N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_avgs, avgs, sizeof(float)*(N-M+1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_stds, stds, sizeof(float)*(N-M+1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lower_bound, lower_bound, sizeof(float)*N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_upper_bound, upper_bound, sizeof(float)*N, cudaMemcpyHostToDevice));

  dim3 dimGrid(grids, 1, 1);
  dim3 dimBlock(blocks, 1, 1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    kernel<<<dimGrid, dimBlock>>>(d_subject, d_avgs, d_stds, d_lower_bound, d_upper_bound, N, M, N-M+1);
    CUDA_CHECK(cudaGetLastError());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  CUDA_CHECK(cudaMemcpy(lb, lb, sizeof(float)*(N-M+1), cudaMemcpyDeviceToHost));

  reference(subject, avgs, stds, lb_h, lower_bound, upper_bound, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabs(lb[i] - lb_h[i]) > 1e-3) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(subject));
  CUDA_CHECK(cudaFree(lower_bound));
  CUDA_CHECK(cudaFree(upper_bound));
  CUDA_CHECK(cudaFree(lb));
  CUDA_CHECK(cudaFree(lb_h));
  CUDA_CHECK(cudaFree(avgs));
  CUDA_CHECK(cudaFree(stds));
  return 0;
}

__global__ void kernel(float *subject, float *avgs, float *stds, float *lower_bound, float *upper_bound, int N, int M, int N_M_1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N_M_1) {
    float residues = 0;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      float value = (subject[idx+i] - avg) / std;
      float lower = value - lower_bound[i];
      float upper = value - upper_bound[i];

      residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
    }

    lb[idx] = residues;
  }
}
```

Note that the `reference` function is assumed to be implemented in the `reference.h` header file, and its implementation is not provided here. Also, the `cudaMalloc` and `cudaMemcpy` calls are used to manage memory on the GPU, and the `kernel` function is a CUDA kernel that is launched on the GPU.