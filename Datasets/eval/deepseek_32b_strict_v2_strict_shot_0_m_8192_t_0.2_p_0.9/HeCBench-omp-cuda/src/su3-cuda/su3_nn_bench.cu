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
#include "lattice.hpp"
#include "mat_nn_openmp.hpp"
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
#include <cuda_runtime.h>
#include "lattice.hpp"

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

unsigned int verbose = 1;
size_t warmups = 1;

int g_argc;
char **g_argv;

template<class T>
bool almost_equal(T x, T y, double tol) {
    if (std::isnan(x) || std::isnan(y))
        return false;
    return std::abs(x - y) < tol;
}

template<class T>
bool almost_equal(std::complex<T> x, std::complex<T> y, double tol) {
    if (std::isnan(x.real()) || std::isnan(x.imag()) || std::isnan(y.real()) || std::isnan(y.imag()))
        return false;
    return std::abs(x - y) < tol;
}

void init_link(su3_matrix *s, Complx val) {
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l)
                s[j].e[k][l] = val;
}

void make_lattice(site *s, size_t n, Complx val) {
    int nx = n, ny = n, nz = n, nt = n;
    for (int t = 0; t < nt; ++t) {
        int i = t * nz * ny * nx;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x, ++i) {
                    s[i].x = x;
                    s[i].y = y;
                    s[i].z = z;
                    s[i].t = t;
                    s[i].index = x + nx * (y + ny * (z + nz * t));
                    s[i].parity = (x + y + z + t) % 2 == 0 ? EVEN : ODD;
                    init_link(&s[i].link[0], val);
                }
    }
}

__device__ void init_link_cuda(su3_matrix *s, Complx val) {
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l)
                s[j].e[k][l] = val;
}

__global__ void su3_mat_nn_kernel(site *a, su3_matrix *b, site *c, size_t total_sites) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_sites) return;

    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                Complx cc = {0.0, 0.0};
                for (int m = 0; m < 3; ++m) {
                    cc += a[idx].link[j].e[k][m] * b[j].e[m][l];
                }
                c[idx].link[j].e[k][l] = cc;
            }
        }
    }
}

double su3_mat_nn(site *a, su3_matrix *b, site *c, size_t total_sites, size_t iterations, size_t threads_per_group, int device) {
    site *a_gpu, *c_gpu;
    su3_matrix *b_gpu;

    cudaSetDevice(device);
    cudaMalloc((void **)&a_gpu, sizeof(site) * total_sites);
    cudaMalloc((void **)&b_gpu, sizeof(su3_matrix) * 4);
    cudaMalloc((void **)&c_gpu, sizeof(site) * total_sites);

    cudaMemcpy(a_gpu, a, sizeof(site) * total_sites, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(su3_matrix) * 4, cudaMemcpyHostToDevice);

    dim3 block(threads_per_group);
    dim3 grid((total_sites + threads_per_group - 1) / threads_per_group);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (size_t w = 0; w < warmups; ++w) {
        su3_mat_nn_kernel<<<grid, block>>>(a_gpu, b_gpu, c_gpu, total_sites);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (size_t i = 0; i < iterations; ++i) {
        su3_mat_nn_kernel<<<grid, block>>>(a_gpu, b_gpu, c_gpu, total_sites);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&ttotal, start, stop);

    cudaMemcpy(c, c_gpu, sizeof(site) * total_sites, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return ttotal / 1000.0;
}

int main(int argc, char **argv) {
    size_t iterations = ITERATIONS;
    size_t ldim = LDIM;
    size_t threads_per_group = 128;
    int device = -1;

    int opt;
    g_argc = argc;
    g_argv = argv;

    while ((opt = getopt(argc, argv, ":hi:l:t:v:d:w:n:")) != -1) {
        switch (opt) {
            case 'i': iterations = atoi(optarg); break;
            case 'l': ldim = atoi(optarg); break;
            case 't': threads_per_group = atoi(optarg); break;
            case 'v': verbose = atoi(optarg); break;
            case 'd': device = atoi(optarg); break;
            case 'w': warmups = atoi(optarg); break;
            case 'h':
                fprintf(stderr, "Usage: %s [-i iterations] [-l lattice dimension] \
[-t threads per workgroup] [-d device] [-v verbosity level [0,1,2,3]] [-w warmups]\n", argv[0]);
                exit(1);
        }
    }

    size_t total_sites = ldim * ldim * ldim * ldim;
    std::vector<site> a(total_sites);
    std::vector<su3_matrix> b(4);
    std::vector<site> c(total_sites);

    make_lattice(a.data(), ldim, Complx{1.0, 0.0});
    init_link(b.data(), Complx{1.0 / 3.0, 0.0});

    if (verbose >= 1) {
        printf("Number of sites = %zu^4\n", ldim);
        printf("Executing %zu iterations with %zu warmups\n", iterations, warmups);
        if (threads_per_group != 0)
            printf("Threads per group = %zu\n", threads_per_group);
    }

    const double ttotal = su3_mat_nn(a.data(), b.data(), c.data(), total_sites, iterations, threads_per_group, device);
    if (verbose >= 1)
        printf("Total kernel execution time = %f (s)\n", ttotal);

    const double tflop = (double)iterations * total_sites * 864.0;
    printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);

    const double memory_usage = (double)sizeof(site) * (a.capacity() + c.capacity()) + sizeof(su3_matrix) * b.capacity();
    printf("Total GByte/s (GPU memory)  = %.3f\n", iterations * memory_usage / ttotal / 1.0e9);
    fflush(stdout);

    for (int i = 0; i < total_sites; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l) {
                    Complx cc = {0.0, 0.0};
                    for (int m = 0; m < 3; ++m) {
                        cc += a[i].link[j].e[k][m] * b[j].e[m][l];
                    }
                    assert(almost_equal(c[i].link[j].e[k][l], cc, 1E-6));
                }

    if (verbose >= 2) {
        printf("Total allocation for matrices = %.3f MiB\n", memory_usage / 1048576.0);
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0)
            printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss / 1024.0);
    }
}