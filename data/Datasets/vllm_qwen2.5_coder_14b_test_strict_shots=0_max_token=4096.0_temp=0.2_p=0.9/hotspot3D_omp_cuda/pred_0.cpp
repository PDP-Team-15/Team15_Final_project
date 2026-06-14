```cpp
#include <sys/types.h>
#include <chrono>
#include <cuda_runtime.h>
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

__global__ void computeTempKernel(float* tIn, float* pIn, float* tOut, int numCols, int numRows, int layers, float Cap, float Rx, float Ry, float Rz, float dt, float amb_temp, float stepDivCap, float ce, float cw, float cn, float cs, float ct, float cb, float cc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < numCols && j < numRows && k < layers)
  {
    int c = i + j * numCols + k * numCols * numRows;
    int xy = numCols * numRows;

    int W = (i == 0) ? c : c - 1;
    int E = (i == numCols - 1) ? c : c + 1;
    int N = (j == 0) ? c : c - numCols;
    int S = (j == numRows - 1) ? c : c + numCols;
    int B = (k == 0) ? c : c - xy;
    int T = (k == layers - 1) ? c : c + xy;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[T];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
      + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;

    for (int l = 1; l < layers - 1; ++l) {
      temp1 = temp2;
      temp2 = temp3;
      temp3 = tIn[B + l * xy];
      tOut[c + l * xy] = cc * temp2 + cw * tIn[W + l * xy] + ce * tIn[E + l * xy] + cs * tIn[S + l * xy]
        + cn * tIn[N + l * xy] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c + l * xy] + ct * amb_temp;
    }

    temp1 = temp2;
    temp2 = temp3;
    tOut[c + (layers - 1) * xy] = cc * temp2 + cw * tIn[W + (layers - 1) * xy] + ce * tIn[E + (layers - 1) * xy] + cs * tIn[S + (layers - 1) * xy]
      + cn * tIn[N + (layers - 1) * xy] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c + (layers - 1) * xy] + ct * amb_temp;
  }
}

int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc, argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile = argv[4];
  tfile = argv[5];
  ofile = argv[6];
  int numCols = atoi(argv[1]);
  int numRows = atoi(argv[1]);
  int layers = atoi(argv[2]);

  float dx = chip_height / numRows;
  float dy = chip_width / numCols;
  float dz = t_chip / layers;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx = dy / (2.0 * K_SI * t_chip * dx);
  float Ry = dx / (2.0 * K_SI * t_chip * dy);
  float Rz = dz / (K_SI * dx * dy);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;
  cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

  int size = numCols * numRows * layers;
  float* tIn = (float*)calloc(size, sizeof(float));
  float* pIn = (float*)calloc(size, sizeof(float));
  float* tCopy = (float*)malloc(size * sizeof(float));
  float* tOut = (float*)calloc(size, sizeof(float));
  float* sel;

  readinput(tIn, numRows, numCols, layers, tfile);
  readinput(pIn, numRows, numCols, layers, pfile);

  memcpy(tCopy, tIn, size * sizeof(float));

  long long start = get_time();

  float* d_tIn, *d_pIn, *d_tOut;
  cudaMalloc(&d_tIn, size * sizeof(float));
  cudaMalloc(&d_pIn, size * sizeof(float));
  cudaMalloc(&d_tOut, size * sizeof(float));

  cudaMemcpy(d_tIn, tIn, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pIn, pIn, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocksPerGrid((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y,
                     (layers + threadsPerBlock.z - 1) / threadsPerBlock.z);

  auto kstart = std::chrono::steady_clock::now();

  for (int j = 0; j < iterations; j++)
  {
    computeTempKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tIn, d_pIn, d_tOut, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, stepDivCap, ce, cw, cn, cs, ct, cb, cc);
    cudaDeviceSynchronize();

    float* temp = d_tIn;
    d_tIn = d_tOut;
    d_tOut = temp;
  }

  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Average kernel execution time %f (us)\n", (ktime * 1e-3f) / iterations);

  cudaMemcpy(tIn, d_tIn, size * sizeof(float), cudaMemcpyDeviceToHost);

  if (iterations & 01) {
    sel = tIn;
  }
  else {
    cudaMemcpy(tOut, d_tOut, size * sizeof(float), cudaMemcpyDeviceToHost);
    sel = tOut;
  }

  cudaFree(d_tIn);
  cudaFree(d_pIn);
  cudaFree(d_tOut);

  long long stop = get_time();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(sel, answer, numRows * numCols * layers);
  float time = (float)((stop - start) / (1000.0 * 1000.0));
  printf("Device offloading time: %.3f (s)\n", time);
  printf("Root-mean-square error: %e\n", acc);

  writeoutput(tOut, numRows, numCols, layers, ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  return 0;
}
```