Here's the CUDA translation of the provided OpenMP code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <complex>
#include <chrono>
typedef std::chrono::system_clock Clock;

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef LDIM
#  define LDIM 32       

#endif
#ifndef PRECISION
#  define PRECISION 2  

#endif

// CUDA device properties
#include <cuda_runtime.h>

// CUDA kernel launch configuration
#define BLOCK_SIZE 128

// CUDA device ID
int device = -1;

// CUDA stream
cudaStream_t stream;

// CUDA event for timing
cudaEvent_t start, stop;

// CUDA device memory allocation
void* d_a, *d_b, *d_c;

// CUDA kernel launch function
void launch_kernel(int iterations, int total_sites, int threads_per_group) {
  // Launch kernel
  su3_mat_nn<<<threads_per_group, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, total_sites, iterations);
  // Synchronize kernel execution
  cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
  size_t iterations = ITERATIONS;
  size_t ldim = LDIM;
  size_t threads_per_group = 128; 

  // Parse command-line arguments
  int opt;
  g_argc = argc;
  g_argv = argv;
  

  while ((opt=getopt(argc, argv, ":hi:l:t:v:d:w:n:")) != -1) {
    switch (opt) {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'l':
      ldim = atoi(optarg);
      break;
    case 't':
      threads_per_group = atoi(optarg);
      break;
    case 'v':
      verbose = atoi(optarg);
      break;
    case 'd':
      device = atoi(optarg);
      break;
    case 'w':
      warmups = atoi(optarg);
      break;
    case 'h':
      fprintf(stderr, "Usage: %s [-i iterations] [-l lattice dimension] \
[-t threads per workgroup] [-d device] [-v verbosity level [0,1,2,3]] [-w warmups]\n", argv[0]);
      exit (1);
    }
  }

  // Initialize CUDA device
  cudaSetDevice(device);
  cudaStreamCreate(&stream);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate host memory
  size_t total_sites = ldim*ldim*ldim*ldim;
  std::vector<site> a(total_sites);
  std::vector<su3_matrix> b(4);
  std::vector<site> c(total_sites);

  // Initialize host memory
  make_lattice(a.data(), ldim, Complx{1.0,0.0});
  init_link(b.data(), Complx{1.0/3.0,0.0});

  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(site)*total_sites);
  cudaMalloc((void**)&d_b, sizeof(su3_matrix)*4);
  cudaMalloc((void**)&d_c, sizeof(site)*total_sites);

  // Copy host memory to device memory
  cudaMemcpy(d_a, a.data(), sizeof(site)*total_sites, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), sizeof(su3_matrix)*4, cudaMemcpyHostToDevice);

  if (verbose >= 1) {
    printf("Number of sites = %zu^4\n", ldim);
    printf("Executing %zu iterations with %zu warmups\n", iterations, warmups);
    if (threads_per_group != 0)
      printf("Threads per group = %zu\n", threads_per_group);
  }

  // Warm-up kernel execution
  for (int i = 0; i < warmups; i++) {
    launch_kernel(iterations, total_sites, threads_per_group);
  }

  // Measure kernel execution time
  cudaEventRecord(start, stream);
  launch_kernel(iterations, total_sites, threads_per_group);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double ttotal = milliseconds / 1000.0;

  if (verbose >= 1)
    printf("Total kernel execution time = %f (s)\n", ttotal);
  

  const double tflop = (double)iterations * total_sites * 864.0;
  printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);

  const double memory_usage = (double)sizeof(site)*(a.capacity()+c.capacity())+sizeof(su3_matrix)*b.capacity();
  printf("Total GByte/s (GPU memory)  = %.3f\n", iterations * memory_usage / ttotal / 1.0e9);
  fflush(stdout);

  // Copy device memory back to host memory
  cudaMemcpy(c.data(), d_c, sizeof(site)*total_sites, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j)  for(int k=0;k<3;++k)  for(int l=0;l<3;++l) {
    Complx cc = {0.0, 0.0};
    for(int m=0;m<3;m++) {
      #ifdef MILC_COMPLEX
        CMULSUM( a[i].link[j].e[k][m], b[j].e[m][l], cc)
      #else
        cc += a[i].link[j].e[k][m] * b[j].e[m][l];
      #endif
    }

    #ifdef MILC_COMPLEX
       assert(almost_equal(c[i].link[j].e[k][l].real, cc.real, 1E-6));
       assert(almost_equal(c[i].link[j].e[k][l].imag, cc.imag, 1E-6));
    #else
       assert(almost_equal(c[i].link[j].e[k][l], cc, 1E-6));
    #endif
  }

  // Clean up
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  if (verbose >= 2) {
    printf("Total allocation for matrices = %.3f MiB\n", memory_usage / 1048576.0);
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
      printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss/1024.0);
  }
}
```

And here's the CUDA kernel code (`su3_mat_nn.cu`):

```cpp
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// CUDA kernel function
__global__ void su3_mat_nn(site* a, su3_matrix* b, site* c, int total_sites, int iterations) {
  // Get global thread ID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if thread is within bounds
  if (idx < total_sites) {
    // Get site and link data
    site site_data = a[idx];
    su3_matrix link_data = b[threadIdx.x];

    // Perform matrix multiplication
    for (int i = 0; i < iterations; i++) {
      Complx cc = {0.0, 0.0};
      for (int m = 0; m < 3; m++) {
        #ifdef MILC_COMPLEX
          CMULSUM( site_data.link[threadIdx.x].e[m][0], link_data.e[m][0], cc)
        #else
          cc += site_data.link[threadIdx.x].e[m][0] * link_data.e[m][0];
        #endif
      }

      #ifdef MILC_COMPLEX
         c[idx].link[threadIdx.x].e[0][0].real = cc.real;
         c[idx].link[threadIdx.x].e[0][0].imag = cc.imag;
      #else
         c[idx].link[threadIdx.x].e[0][0] = cc;
      #endif
    }
  }
}
```

Note that this code assumes that the `site` and `su3_matrix` structs are defined in the `lattice.hpp` header file, and that the `CMULSUM` macro is defined in the `mat_nn_openmp.hpp` header file. You may need to modify the code to match your specific implementation.