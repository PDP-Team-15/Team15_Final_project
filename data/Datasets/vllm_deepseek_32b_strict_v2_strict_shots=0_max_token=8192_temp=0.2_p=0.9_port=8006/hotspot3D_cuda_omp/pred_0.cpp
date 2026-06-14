<think>
Okay, so I need to translate the given CUDA code into OpenMP (OMP). Let me start by understanding what the CUDA code is doing. 

First, looking at the 3D.cu file, it's a program that simulates heat distribution in a 3D chip. It uses CUDA for parallel processing. The main function reads input files, sets up some parameters, allocates memory, copies data to the GPU, runs the kernel multiple times, and then copies the results back to the CPU. There's also a helper function computeTempCPU which does the same computation but on the CPU for comparison.

The hotspot3d kernel is where the computation happens. It's a 2D grid of threads, each handling a cell in the 3D grid. Each thread computes the temperature for its cell based on neighboring cells and power input.

Now, to translate this to OpenMP, I need to replace the CUDA-specific parts with OpenMP directives and functions. 

Starting with the includes. The CUDA headers like cuda.h and cuda_runtime.h are no longer needed. Instead, I'll include omp.h for OpenMP. Also, since the code uses std::chrono for timing, I'll include chrono.

Next, the kernel function hotspot3d is a CUDA __global__ function. In OpenMP, this will become a regular function, but with parallel directives. However, since OpenMP doesn't support nested parallelism well, I might need to adjust how I structure the loops.

Looking at the main function, the CUDA setup includes allocating device memory with cudaMalloc and copying data with cudaMemcpy. In OpenMP, all computations are done on the CPU, so I can remove the device memory allocation and just use host memory.

The kernel launch in CUDA is done with <<<gridDim, blockDim>>>. In OpenMP, I'll replace this with #pragma omp parallel for directives to parallelize the loops. 

The hotspot3d function in CUDA is called for each iteration. In OpenMP, I'll loop over the iterations and within each iteration, use parallel loops to compute each cell's temperature.

I also need to handle the swapping of tIn and tOut arrays. In CUDA, this is done by swapping pointers, which is the same in OpenMP.

The helper functions like readinput, writeoutput, and computeTempCPU are already in the helper file. Since computeTempCPU is a CPU function, it can remain as is, but I might need to adjust it for OpenMP if necessary.

Now, considering the data structures and variables. The CUDA code uses pointers to device memory, which I can replace with regular host pointers. The size variables and loops remain the same, but I'll add OpenMP directives to parallelize the loops.

For the hotspot3d function, I'll convert it into a function that can be called within an OpenMP parallel region. However, since OpenMP doesn't support the same level of thread management as CUDA, I'll have to structure the loops differently. Each thread in OpenMP will handle a range of cells, similar to how CUDA blocks and threads handle grid cells.

I'll also need to manage shared variables carefully. In CUDA, each thread has its own registers, but in OpenMP, variables need to be properly scoped to avoid race conditions. I'll use private variables where necessary.

Timing is another consideration. The CUDA code uses cudaDeviceSynchronize and std::chrono for timing. In OpenMP, I can use std::chrono directly since all computations are on the CPU.

Error checking in CUDA is done with functions like cudaMalloc and cudaMemcpy. In OpenMP, I can remove these checks since we're using host memory and standard memory functions like malloc and memcpy.

Finally, I'll ensure that all CUDA-specific syntax is removed and replaced with OpenMP directives. This includes removing __global__ and __device__ qualifiers, replacing kernel launches with parallel loops, and managing memory as per host requirements.

Putting it all together, the translated code will have the same structure but with CUDA parts replaced by OpenMP directives and host memory management. The main function will set up the data, run the computations in parallel using OpenMP, and then output the results.
</think>

```cpp
#include <sys/types.h>
#include <chrono>
#include <omp.h>
#include "3D_helper.h"

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

#define FACTOR_CHIP  0.5

float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");
  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}

void hotspot3d(
    const float* tIn, 
    const float* pIn, 
          float* tOut,
    const int numCols, 
    const int numRows, 
    const int layers,
    const float ce, 
    const float cw,
    const float cn, 
    const float cs,
    const float ct,
    const float cb,
    const float cc,
    const float stepDivCap)
{
  float amb_temp = 80.0;

  int i, j, c, xy = numCols * numRows;
  int W, E, N, S;

  #pragma omp parallel for private(i, j, c, W, E, N, S)
  for (i = 0; i < numCols; ++i) {
    for (j = 0; j < numRows; ++j) {
      c = i + j * numCols;
      W = (i == 0) ? c : c - 1;
      E = (i == numCols-1) ? c : c + 1;
      N = (j == 0) ? c : c - numCols;
      S = (j == numRows-1) ? c : c + numCols;

      float temp1, temp2, temp3;
      temp1 = temp2 = tIn[c];
      temp3 = tIn[c+xy];
      tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    }
  }

  #pragma omp parallel for private(i, j, c, W, E, N, S)
  for (int k = 1; k < layers-1; ++k) {
    for (i = 0; i < numCols; ++i) {
      for (j = 0; j < numRows; ++j) {
        c = i + j * numCols + k * xy;
        W = c - 1;
        E = c + 1;
        N = c - numCols;
        S = c + numCols;

        float temp1, temp2, temp3;
        temp1 = tIn[c - xy];
        temp2 = tIn[c - xy + xy];
        temp3 = tIn[c + xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
          + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
      }
    }
  }

  #pragma omp parallel for private(i, j, c, W, E, N, S)
  for (i = 0; i < numCols; ++i) {
    for (j = 0; j < numRows; ++j) {
      c = i + j * numCols + (layers-1) * xy;
      W = (i == 0) ? c : c - 1;
      E = (i == numCols-1) ? c : c + 1;
      N = (j == 0) ? c : c - numCols;
      S = (j == numRows-1) ? c : c + numCols;

      float temp1, temp2, temp3;
      temp1 = tIn[c - xy];
      temp2 = tIn[c];
      temp3 = tIn[c + xy];
      tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc,argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw = stepDivCap/ Rx;
  cn               = cs = stepDivCap/ Ry;
  ct               = cb = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int size = numCols * numRows * layers;
  float* tIn   = (float*) calloc(size,sizeof(float));
  float* pIn   = (float*) calloc(size,sizeof(float));
  float* tCopy = (float*) malloc(size * sizeof(float));
  float* tOut  = (float*) calloc(size,sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  memcpy(tCopy,tIn, size * sizeof(float));

  long long start = get_time();

  for(int j = 0; j < iterations; j++)
  {
    hotspot3d(tIn, pIn, tOut, numCols, numRows, layers,
              ce, cw, cn, cs, ct, cb, cc, stepDivCap);

    float* temp = tIn;
    tIn = tOut;
    tOut = temp;
  }

  long long stop = get_time();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Total execution time: %.3f (s)\n",time);
  printf("Root-mean-square error: %e\n",acc);

  writeoutput(tOut,numRows,numCols,layers,ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  return 0;
}
```