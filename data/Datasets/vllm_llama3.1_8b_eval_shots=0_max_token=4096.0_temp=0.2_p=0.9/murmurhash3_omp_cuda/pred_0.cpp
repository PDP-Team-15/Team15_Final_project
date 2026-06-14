Here's the CUDA version of the provided code:

```cpp
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define  FORCE_INLINE inline __attribute__((always_inline))

inline uint64_t rotl64 ( uint64_t x, int8_t r )
{
  return (x << r) | (x >> (64 - r));
}

#define ROTL64(x,y)  rotl64(x,y)

#define BIG_CONSTANT(x) (x##LLU)

FORCE_INLINE uint64_t getblock64 ( const uint8_t * p, uint32_t i )
{
  uint64_t s = 0;
  for (uint32_t n = 0; n < 8; n++) {
    s |= ((uint64_t)p[8*i+n] << (n*8));
  }
  return s;
}

FORCE_INLINE uint64_t fmix64 ( uint64_t k )
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

__global__ void MurmurHash3_x64_128_kernel(const uint8_t * key, const uint32_t len,
                                            const uint32_t seed, uint64_t * out)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len / 16) return;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  uint64_t k1 = getblock64(key, idx*2+0);
  uint64_t k2 = getblock64(key, idx*2+1);

  k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;

  h1 = ROTL64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

  k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;

  h2 = ROTL64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
}

int main(int argc, char** argv) 
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  } 
  uint32_t numKeys = atoi(argv[1]);
  uint32_t repeat = atoi(argv[2]);

  srand(3);
  uint32_t i;
  

  uint32_t* length = (uint32_t*) malloc (sizeof(uint32_t) * numKeys);
  

  uint8_t** keys = (uint8_t**) malloc (sizeof(uint8_t*) * numKeys);
  

  uint64_t** out = (uint64_t**) malloc (sizeof(uint64_t*) * numKeys);

  for (i = 0; i < numKeys; i++) {
    length[i] = rand() % 10000;
    keys[i] = (uint8_t*) malloc (length[i]);
    out[i] = (uint64_t*) malloc (2*sizeof(uint64_t));
    for (uint32_t c = 0; c < length[i]; c++) {
      keys[i][c] = c % 256;
    }
    MurmurHash3_x64_128 (keys[i], length[i], i, out[i]);
#ifdef DEBUG
    printf("%lu %lu\n", out[i][0], out[i][1]);
#endif
  }

  

  

  

  uint64_t* d_out = (uint64_t*) malloc (sizeof(uint64_t) * 2 * numKeys);
  uint32_t* d_length = (uint32_t*) malloc (sizeof(uint32_t) * (numKeys+1));

  

  uint32_t total_length = 0;
  d_length[0] = 0;
  for (uint32_t i = 0; i < numKeys; i++) {
    total_length += length[i];
    d_length[i+1] = total_length;
  }

  

  uint8_t* d_keys = (uint8_t*) malloc (sizeof(uint8_t) * total_length);
  for (uint32_t i = 0; i < numKeys; i++) {
    memcpy(d_keys+d_length[i], keys[i], length[i]);
  }
  

  for (uint32_t i = 0; i < numKeys; i++) {
    assert (0 == memcmp(d_keys+d_length[i], keys[i], length[i]));
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMalloc((void**)&d_out, sizeof(uint64_t) * 2 * numKeys);
  cudaMalloc((void**)&d_length, sizeof(uint32_t) * (numKeys+1));
  cudaMalloc((void**)&d_keys, sizeof(uint8_t) * total_length);

  cudaMemcpy(d_length, length, sizeof(uint32_t) * numKeys, cudaMemcpyHostToDevice);
  cudaMemcpy(d_keys, keys[0], sizeof(uint8_t) * length[0], cudaMemcpyHostToDevice);

  for (uint32_t i = 1; i < numKeys; i++) {
    cudaMemcpy(d_keys + d_length[i-1], keys[i], sizeof(uint8_t) * length[i], cudaMemcpyHostToDevice);
  }

  dim3 blockSize(BLOCK_SIZE);
  dim3 gridSize((numKeys + blockSize.x - 1) / blockSize.x);

  cudaEventRecord(start, 0);
  for (uint32_t n = 0; n < repeat; n++) {
    MurmurHash3_x64_128_kernel<<<gridSize, blockSize>>>(d_keys, d_length[numKeys], numKeys, d_out);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Average kernel execution time %f (ms)\n", milliseconds / repeat);

  cudaMemcpy(out[0], d_out, sizeof(uint64_t) * 2, cudaMemcpyDeviceToHost);

  bool error = false;
  for (uint32_t i = 0; i < numKeys; i++) {
    if (out[i][0] != d_out[2*i] ||  out[i][1] != d_out[2*i+1]) {
      error = true;
      break;
    }
  }
  if (error) printf("FAIL\n");
  else printf("SUCCESS\n");

  for (uint32_t i = 0; i < numKeys; i++) {
    free(out[i]);
    free(keys[i]);
  }
  free(keys);
  free(out);
  free(length);
  free(d_keys);
  free(d_out);
  free(d_length);
  cudaFree(d_out);
  cudaFree(d_length);
  cudaFree(d_keys);
  return 0;
}
```

Note that the CUDA version uses CUDA events to measure the execution time of the kernel, and the `MurmurHash3_x64_128_kernel` function is launched using the `<<<gridSize, blockSize>>>` syntax. The host memory is copied to device memory using `cudaMemcpy`, and the results are copied back to host memory using `cudaMemcpy`. The CUDA runtime is used to manage the memory and launch the kernel.