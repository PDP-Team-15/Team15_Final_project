#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "utils.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include "utils.h"
#include "kernel.h"

#define CUDA_CHECK(x) do { if((x)!=cudaSuccess) { \
    printf("CUDA Error: %s (%d)\n", cudaGetErrorString(x), __LINE__); \
    exit(1); }} while(0)

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  float *th, *pii, *q;
  float *qc, *qi, *qr, *qs;
  float *den, *p, *delz;
  float *rain,*rainncv;
  float *sr;
  float *snow, *snowncv;

  float delt = 10.f;
  int ims = 0, ime = 59, jms = 0, jme = 45, kms = 0, kme = 2;
  int ips = 0, ipe = 59, jps = 0, jpe = 45, kps = 0, kpe = 2;
  int d3 = (ime-ims+1) * (jme-jms+1) * (kme-kms+1) ;
  int d2 = (ime-ims+1) * (jme-jms+1) ;

  int dips = 0 ; int dipe = (ipe-ips+1) ;
  int djps = 0 ; int djpe = (jpe-jps+1) ;
  int dkps = 0 ; int dkpe = (kpe-kps+1) ;

  float rain_sum = 0, snow_sum = 0;

  long time = 0;
  for (int i = 0; i < repeat; i++) {
    // Allocate memory on host
    cudaMalloc(&th, d3 * sizeof(float));
    cudaMalloc(&pii, d3 * sizeof(float));
    cudaMalloc(&q, d3 * sizeof(float));
    cudaMalloc(&qc, d3 * sizeof(float));
    cudaMalloc(&qi, d3 * sizeof(float));
    cudaMalloc(&qr, d3 * sizeof(float));
    cudaMalloc(&qs, d3 * sizeof(float));
    cudaMalloc(&den, d3 * sizeof(float));
    cudaMalloc(&p, d3 * sizeof(float));
    cudaMalloc(&delz, d3 * sizeof(float));
    cudaMalloc(&rain, d2 * sizeof(float));
    cudaMalloc(&rainncv, d2 * sizeof(float));
    cudaMalloc(&sr, d2 * sizeof(float));
    cudaMalloc(&snow, d2 * sizeof(float));
    cudaMalloc(&snowncv, d2 * sizeof(float));

    int remx = (ipe-ips+1) % XXX != 0 ? 1 : 0 ;
    int remy = (jpe-jps+1) % YYY != 0 ? 1 : 0 ;

    const int teamX = (ipe-ips+1) / XXX + remx;
    const int teamY = (jpe-jps+1) / YYY + remy;

    dim3 blockSize(XXX, YYY, 1);
    dim3 gridSize(teamX, teamY, 1);

    auto start = std::chrono::steady_clock::now();

    wsm<<<gridSize, blockSize>>>(
        th, pii, q, qc, qi, qr, qs, den, p, delz,
        rain, rainncv,
        sr,
        snow, snowncv,
        delt,
        dips+1 , (ipe-ips+1), 
        djps+1 , (jpe-jps+1), 
        dkps+1 , (kpe-kps+1),  
        dips+1 , dipe,        
        djps+1 , djpe,        
        dkps+1 , dkpe,        
        dips+1 , dipe,        
        djps+1 , djpe,        
        dkps+1 , dkpe,        
        teamX , teamY);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(rain, rain, d2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(snow, snow, d2 * sizeof(float), cudaMemcpyDeviceToHost));
  }

  rain_sum = snow_sum = 0;
  for (int i = 0; i < d2; i++) {
    rain_sum += rain[i];
    snow_sum += snow[i];
  }

  // Cleanup
  cudaFree(th);
  cudaFree(pii);
  cudaFree(q);
  cudaFree(qc);
  cudaFree(qi);
  cudaFree(qr);
  cudaFree(qs);
  cudaFree(den);
  cudaFree(p);
  cudaFree(delz);
  cudaFree(rain);
  cudaFree(rainncv);
  cudaFree(sr);
  cudaFree(snow);
  cudaFree(snowncv);

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);
  return(0);
}