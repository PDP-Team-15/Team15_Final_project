#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>

#define BLOCK_SIZE 256

#define FORCE_INLINE __forceinline__

__device__ uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define ROTL64(x, y) rotl64(x, y)

#define BIG_CONSTANT(x) (x##LLU)

__device__ uint64_t getblock64(const uint8_t *p, uint32_t i) {
  uint64_t s = 0;
  for (uint32_t n = 0; n < 8; n++) {
    s |= ((uint64_t)p[8 * i + n] << (n * 8));
  }
  return s;
}

__device__ uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

__global__ void MurmurHash3_x64_128_kernel(const uint8_t *key, const uint32_t *length, const uint32_t *seed, uint64_t *out, uint32_t numKeys) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numKeys) return;

  const uint8_t *data = &key[length[tid]];
  uint64_t h1 = seed[tid];
  uint64_t h2 = seed[tid];

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  uint32_t nblocks = length[tid] / 16;

  for (uint32_t i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock64(data, i * 2 + 0);
    uint64_t k2 = getblock64(data, i * 2 + 1);

    k1 *= c1; k1 = ROTL64(k1, 31); k1 *= c2; h1 ^= k1;

    h1 = ROTL64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;

    k2 *= c2; k2 = ROTL64(k2, 33); k2 *= c1; h2 ^= k2;

    h2 = ROTL64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
  }

  const uint8_t *tail = &data[nblocks * 16];

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (length[tid] & 15) {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48;
    case 14: k2 ^= ((uint64_t)tail[13]) << 40;
    case 13: k2 ^= ((uint64_t)tail[12]) << 32;
    case 12: k2 ^= ((uint64_t)tail[11]) << 24;
    case 11: k2 ^= ((uint64_t)tail[10]) << 16;
    case 10: k2 ^= ((uint64_t)tail[9]) << 8;
    case 9: k2 ^= ((uint64_t)tail[8]) << 0;
       k2 *= c2; k2 = ROTL64(k2, 33); k2 *= c1; h2 ^= k2;

    case 8: k1 ^= ((uint64_t)tail[7]) << 56;
    case 7: k1 ^= ((uint64_t)tail[6]) << 48;
    case 6: k1 ^= ((uint64_t)tail[5]) << 40;
    case 5: k1 ^= ((uint64_t)tail[4]) << 32;
    case 4: k1 ^= ((uint64_t)tail[3]) << 24;
    case 3: k1 ^= ((uint64_t)tail[2]) << 16;
    case 2: k1 ^= ((uint64_t)tail[1]) << 8;
    case 1: k1 ^= ((uint64_t)tail[0]) << 0;
       k1 *= c1; k1 = ROTL64(k1, 31); k1 *= c2; h1 ^= k1;
  };

  h1 ^= length[tid]; h2 ^= length[tid];

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  out[tid * 2] = h1;
  out[tid * 2 + 1] = h2;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <number of keys> <repeat>" << std::endl;
    return 1;
  }
  uint32_t numKeys = std::atoi(argv[1]);
  uint32_t repeat = std::atoi(argv[2]);

  srand(3);
  uint32_t i;

  uint32_t *h_length = new uint32_t[numKeys];
  uint8_t **h_keys = new uint8_t *[numKeys];
  uint64_t **h_out = new uint64_t *[numKeys];

  for (i = 0; i < numKeys; i++) {
    h_length[i] = rand() % 10000;
    h_keys[i] = new uint8_t[h_length[i]];
    h_out[i] = new uint64_t[2];
    for (uint32_t c = 0; c < h_length[i]; c++) {
      h_keys[i][c] = c % 256;
    }
    MurmurHash3_x64_128(h_keys[i], h_length[i], i, h_out[i]);
#ifdef DEBUG
    std::cout << h_out[i][0] << " " << h_out[i][1] << std::endl;
#endif
  }

  uint64_t *d_out = nullptr;
  uint32_t *d_length = nullptr;
  uint8_t *d_keys = nullptr;

  cudaMalloc((void **)&d_out, sizeof(uint64_t) * 2 * numKeys);
  cudaMalloc((void **)&d_length, sizeof(uint32_t) * (numKeys + 1));
  cudaMalloc((void **)&d_keys, sizeof(uint8_t) * 10000 * numKeys);

  cudaMemcpy(d_length, h_length, sizeof(uint32_t) * (numKeys + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_keys, h_keys, sizeof(uint8_t) * 10000 * numKeys, cudaMemcpyHostToDevice);

  uint32_t *d_seed = new uint32_t[numKeys];
  for (i = 0; i < numKeys; i++) {
    d_seed[i] = i;
  }

  uint32_t blocksPerGrid = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

  {
    auto start = std::chrono::steady_clock::now();

    for (uint32_t n = 0; n < repeat; n++) {
      MurmurHash3_x64_128_kernel<<<blocksPerGrid, BLOCK_SIZE>>>(d_keys, d_length, d_seed, d_out, numKeys);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / repeat << " (s)" << std::endl;
  }

  bool error = false;
  for (i = 0; i < numKeys; i++) {
    if (d_out[i * 2] != h_out[i][0] || d_out[i * 2 + 1] != h_out[i][1]) {
      error = true;
      break;
    }
  }
  if (error) std::cout << "FAIL" << std::endl;
  else std::cout << "SUCCESS" << std::endl;

  for (i = 0; i < numKeys; i++) {
    delete[] h_out[i];
    delete[] h_keys[i];
  }
  delete[] h_out;
  delete[] h_keys;
  delete[] h_length;
  delete[] d_seed;

  cudaFree(d_out);
  cudaFree(d_length);
  cudaFree(d_keys);

  return 0;
}