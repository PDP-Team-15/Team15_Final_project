You are an HPC expert specializing in translating between parallel programming APIs. Output only the translated code inside a ```cpp ... ``` code block. No explanations, no comments outside the code block, no omissions.
For each kernel code provided, translate it from cuda to omp. Output ONLY the complete translated code wrapped in ```cpp ... ```. Do not truncate. Do not modify main(). All function names must match.
Translate the following code from cuda to omp. Output ONLY the complete translated code wrapped in ```cpp ... ```. Do not truncate. Do not modify main(). All function names must match. All functions, macros, and variables used in the output must be fully defined. Definitions must appear before their first use.
Translate the following code  from cuda to omp:
stencil_1d.cu:



#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__
void stencil_1d(const int *__restrict__ in, int *__restrict__ out)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  

  temp[lindex] = in[gindex];

  

  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  

  __syncthreads();

  

  int result = 0;
  for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
    result += temp[lindex + offset];

  

  out[gindex] = result; 
}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int size = length * sizeof(int);
  int pad_size = (length + RADIUS) * sizeof(int);

  int *a, *b;
  

  a = (int *)malloc(pad_size); 
  b = (int *)malloc(size);

  for (int i = 0; i < length+RADIUS; i++) a[i] = i;

  int *d_a, *d_b;
  

  cudaMalloc((void **)&d_a, pad_size);
  cudaMalloc((void **)&d_b, size);

  

  cudaMemcpy(d_a, a, pad_size, cudaMemcpyHostToDevice);

  dim3 grids (length/BLOCK_SIZE);
  dim3 blocks (BLOCK_SIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  

  for (int i = 0; i < repeat; i++)
    stencil_1d <<< grids, blocks >>> (d_a, d_b);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  

  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  

  bool ok = true;
  for (int i = 0; i < 2*RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i+2*RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2*RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i-RADIUS; j <= i+RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  

  free(a);
  free(b); 
  cudaFree(d_a); 
  cudaFree(d_b); 
  return 0;
}
