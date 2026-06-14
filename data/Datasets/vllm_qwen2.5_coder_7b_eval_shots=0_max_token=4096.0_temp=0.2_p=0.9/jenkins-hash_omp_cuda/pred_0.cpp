Here's the CUDA version of the provided OpenMP code:

```cpp
#include <stdio.h>      
#include <stdlib.h>     
#include <string.h>     
#include <chrono>
#include <cuda_runtime.h>

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

__global__ void mixRemainderKernel(unsigned int *a, 
                                   unsigned int *b, 
                                   unsigned int *c, 
                                   unsigned int *k0,
                                   unsigned int *k1,
                                   unsigned int *k2,
                                   unsigned int *length, 
                                   unsigned int *out, 
                                   int N) 
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;

  unsigned int l = length[id];
  unsigned int k0_val = k0[id];
  unsigned int k1_val = k1[id];
  unsigned int k2_val = k2[id];

  switch(l)
  {
    case 12: c[id] += k2_val; b[id] += k1_val; a[id] += k0_val; break;
    case 11: c[id] += k2_val & 0xffffff; b[id] += k1_val; a[id] += k0_val; break;
    case 10: c[id] += k2_val & 0xffff; b[id] += k1_val; a[id] += k0_val; break;
    case 9 : c[id] += k2_val & 0xff; b[id] += k1_val; a[id] += k0_val; break;
    case 8 : b[id] += k1_val; a[id] += k0_val; break;
    case 7 : b[id] += k1_val & 0xffffff; a[id] += k0_val; break;
    case 6 : b[id] += k1_val & 0xffff; a[id] += k0_val; break;
    case 5 : b[id] += k1_val & 0xff; a[id] += k0_val; break;
    case 4 : a[id] += k0_val; break;
    case 3 : a[id] += k0_val & 0xffffff; break;
    case 2 : a[id] += k0_val & 0xffff; break;
    case 1 : a[id] += k0_val & 0xff; break;
    case 0 : out[id] = c[id]; return;              
  }

  final(a[id], b[id], c[id]);
  out[id] = c[id];
}

__global__ void hashlittleKernel(const unsigned int *keys, 
                                 const unsigned int *lens, 
                                 const unsigned int *initvals, 
                                 unsigned int *out, 
                                 int N, 
                                 int block_size) 
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;

  unsigned int a, b, c;                                          
  a = b = c = 0xdeadbeef + ((unsigned int)lens[id]) + initvals[id];

  const unsigned int *k = keys + id * 16;         

  while (lens[id] > 12)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a, b, c);
    lens[id] -= 12;
    k += 3;
  }

  unsigned int r0 = k[0];
  unsigned int r1 = k[1];
  unsigned int r2 = k[2];

  out[id] = mixRemainder(a, b, c, r0, r1, r2, lens[id]);
}

unsigned int hashlittle(const void *key, size_t length, unsigned int initval)
{
  unsigned int a, b, c;                                          

  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  const unsigned int *k = (const unsigned int *)key;         

  while (length > 12)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a, b, c);
    length -= 12;
    k += 3;
  }

  switch(length)
  {
    case 12: c += k[2]; b += k[1]; a += k[0]; break;
    case 11: c += k[2] & 0xffffff; b += k[1]; a += k[0]; break;
    case 10: c += k[2] & 0xffff; b += k[1]; a += k[0]; break;
    case 9 : c += k[2] & 0xff; b += k[1]; a += k[0]; break;
    case 8 : b += k[1]; a += k[0]; break;
    case 7 : b += k[1] & 0xffffff; a += k[0]; break;
    case 6 : b += k[1] & 0xffff; a += k[0]; break;
    case 5 : b += k[1] & 0xff; a += k[0]; break;
    case 4 : a += k[0]; break;
    case 3 : a += k[0] & 0xffffff; break;
    case 2 : a += k[0] & 0xffff; break;
    case 1 : a += k[0] & 0xff; break;
    case 0 : return c;              
  }

  final(a, b, c);
  return c;
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

  unsigned int *d_keys, *d_lens, *d_initvals, *d_out, *h_out;

  cudaMalloc((void**)&d_keys, sizeof(unsigned int)*N*16);
  cudaMalloc((void**)&d_lens, sizeof(unsigned int)*N);
  cudaMalloc((void**)&d_initvals, sizeof(unsigned int)*N);
  cudaMalloc((void**)&d_out, sizeof(unsigned int)*N);
  h_out = (unsigned int*)malloc(sizeof(unsigned int)*N);

  srand(2);
  char src[64];
  memcpy(src, str, 64);
  for (unsigned long i = 0; i < N; i++) {
    memcpy((unsigned char*)d_keys+i*16*sizeof(unsigned int), src, 64);
    d_lens[i] = rand()%61;
    d_initvals[i] = i%2;
  }

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    hashlittleKernel<<<(N + block_size - 1) / block_size, block_size>>>(d_keys, d_lens, d_initvals, d_out, N, block_size);
  }

  cudaMemcpy(h_out, d_out, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

  printf("Verify the results computed on the device..\n");
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    c = hashlittle(&d_keys[i*16], d_lens[i], d_initvals[i]);
    if (h_out[i] != c) {
      printf("Error: at %lu gpu hash is %.8x  cpu hash is %.8x\n", i, h_out[i], c);
      error = true;
      break;
    }
  }

  printf("%s\n", error ? "FAIL" : "PASS");

  free(d_keys);
  free(d_lens);
  free(d_initvals);
  free(d_out);
  free(h_out);

  return 0;
}
```

This CUDA code translates the original OpenMP code into a CUDA kernel-based implementation. The `hashlittle` function remains unchanged as it does not contain any parallelizable operations. The `mixRemainder` function has been converted into a CUDA kernel (`mixRemainderKernel`) that operates on arrays of values rather than individual variables. The `hashlittleKernel` function orchestrates the computation by calling `mixRemainderKernel`. The main function sets up the input data, launches the kernels, measures the execution time, and verifies the results.