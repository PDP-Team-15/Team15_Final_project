```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include "utils.h"
#include "kernel.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

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
    ALLOC3(th) ;
    ALLOC3(pii) ;
    ALLOC3(q) ;
    ALLOC3(qc) ;
    ALLOC3(qi) ;
    ALLOC3(qr) ;
    ALLOC3(qs) ;
    ALLOC3(den) ;
    ALLOC3(p) ;
    ALLOC3(delz) ;
    ALLOC2(rain) ;
    ALLOC2(rainncv) ;
    ALLOC2(sr) ;
    ALLOC2(snow) ;
    ALLOC2(snowncv) ;

    int remx = (ipe-ips+1) % XXX != 0 ? 1 : 0 ;
    int remy = (jpe-jps+1) % YYY != 0 ? 1 : 0 ;

    const int teamX = (ipe-ips+1) / XXX + remx;
    const int teamY = (jpe-jps+1) / YYY + remy;

    cudaMallocHost((void**)&th, d3 * sizeof(float));
    cudaMallocHost((void**)&pii, d3 * sizeof(float));
    cudaMallocHost((void**)&q, d3 * sizeof(float));
    cudaMallocHost((void**)&qc, d3 * sizeof(float));
    cudaMallocHost((void**)&qi, d3 * sizeof(float));
    cudaMallocHost((void**)&qr, d3 * sizeof(float));
    cudaMallocHost((void**)&qs, d3 * sizeof(float));
    cudaMallocHost((void**)&den, d3 * sizeof(float));
    cudaMallocHost((void**)&p, d3 * sizeof(float));
    cudaMallocHost((void**)&delz, d3 * sizeof(float));
    cudaMallocHost((void**)&rainncv, d2 * sizeof(float));
    cudaMallocHost((void**)&snowncv, d2 * sizeof(float));
    cudaMallocHost((void**)&sr, d2 * sizeof(float));
    cudaMallocHost((void**)&rain, d2 * sizeof(float));
    cudaMallocHost((void**)&snow, d2 * sizeof(float));

    cudaMemset(th, 0, d3 * sizeof(float));
    cudaMemset(pii, 0, d3 * sizeof(float));
    cudaMemset(q, 0, d3 * sizeof(float));
    cudaMemset(qc, 0, d3 * sizeof(float));
    cudaMemset(qi, 0, d3 * sizeof(float));
    cudaMemset(qr, 0, d3 * sizeof(float));
    cudaMemset(qs, 0, d3 * sizeof(float));
    cudaMemset(den, 0, d3 * sizeof(float));
    cudaMemset(p, 0, d3 * sizeof(float));
    cudaMemset(delz, 0, d3 * sizeof(float));
    cudaMemset(rainncv, 0, d2 * sizeof(float));
    cudaMemset(snowncv, 0, d2 *