#include <stdio.h>      
#include <stdlib.h>     
#include <string.h>     
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

__device__ unsigned int mixRemainder(unsigned int a, 
    unsigned int b, 
    unsigned int c, 
    unsigned int k0,
    unsigned int k1,
    unsigned int k2,
    unsigned int length ) 
{
  switch(length)
  {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff; b+=k1; a+=k0; break;
    case 9 : c+=k2&0xff; b+=k1; a+=k0; break;
    case 8 : b+=k1; a+=k0; break;
    case 7 : b+=k1&0xffffff; a+=k0; break;
    case 6 : b+=k1&0xffff; a+=k0; break;
    case 5 : b+=k1&0xff; a+=k0; break;
    case 4 : a+=k0; break;
    case 3 : a+=k0&0xffffff; break;
    case 2 : a+=k0&0xffff; break;
    case 1 : a+=k0&0xff; break;
    case 0 : return c;              
  }

  final(a,b,c);
  return c;
}

__global__ void hashlittleKernel(const unsigned int *keys, const unsigned int *lens, const unsigned int *initvals, unsigned int *out, size_t N) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= N) return;

  unsigned int length = lens[tid];
  const unsigned int initval = initvals[tid];
  const unsigned int *k = &keys[tid*16];  

  unsigned int a,b,c; 
  unsigned int r0,r1,r2;
  a = b = c = 0xdeadbeef + length + initval;

  while (length > 12) {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a,b,c);
    length -= 12;
    k += 3;
  }
  r0 = k[0];
  r1 = k[1];
  r2 = k[2];

  out[tid] = mixRemainder(a, b, c, r0, r1, r2, length);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <block size> <number of strings> <repeat>\n", argv[0]);
    return 1;
  }

  int block_size = atoi(argv[1]);  
  unsigned long N = atol(argv[2]); 
  int repeat = atoi(argv[3]);

  const char* str = "Four score and seven years ago";
  unsigned int c = hashlittle(str, 30, 1);
  printf("input string: %s hash is %.8x\n", str, c);   

  unsigned int *d_keys, *d_lens, *d_initvals, *d_out;
  unsigned int *h_keys = new unsigned int[N*16];
  unsigned int *h_lens = new unsigned int[N];
  unsigned int *h_initvals = new unsigned int[N];
  unsigned int *h_out = new unsigned int[N];

  srand(2);
  char src[64];
  memcpy(src, str, 64);
  for (unsigned long i = 0; i < N; i++) {
    memcpy((unsigned char*)h_keys+i*16*sizeof(unsigned int), src, 64);
    h_lens[i] = rand()%61;
    h_initvals[i] = i%2;
  }

  cudaMalloc((void**)&d_keys, sizeof(unsigned int)*N*16);
  cudaMalloc((void**)&d_lens, sizeof(unsigned int)*N);
  cudaMalloc((void**)&d_initvals, sizeof(unsigned int)*N);
  cudaMalloc((void**)&d_out, sizeof(unsigned int)*N);

  cudaMemcpy(d_keys, h_keys, sizeof(unsigned int)*N*16, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lens, h_lens, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_initvals, h_initvals, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    hashlittleKernel<<<(N + block_size - 1) / block_size, block_size>>>(d_keys, d_lens, d_initvals, d_out, N);
  }

  cudaMemcpy(h_out, d_out, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

  printf("Verify the results computed on the device..\n");
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    c = hashlittle(&h_keys[i*16], h_lens[i], h_initvals[i]);
    if (h_out[i] != c) {
      printf("Error: at %lu gpu hash is %.8x  cpu hash is %.8x\n", i, h_out[i], c);
      error = true;
      break;
    }
  }

  printf("%s\n", error ? "FAIL" : "PASS");

  delete[] h_keys;
  delete[] h_lens;
  delete[] h_initvals;
  delete[] h_out;

  cudaFree(d_keys);
  cudaFree(d_lens);
  cudaFree(d_initvals);
  cudaFree(d_out);

  return 0;
}