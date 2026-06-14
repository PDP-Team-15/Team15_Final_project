#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <fstream>

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f

__global__ void kernel_videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
    // Implementation of videoSequence kernel
}

__global__ void kernel_particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
    // Implementation of particleFilter kernel
}

int main(int argc, char * argv[]) {
    // Main function remains unchanged
}