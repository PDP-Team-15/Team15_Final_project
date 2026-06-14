```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include "utils.h"
#include "kernel.h"

#define BLOCK_X 16
#define BLOCK_Y 16

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

    cudaMalloc((void**)&th, d3 * sizeof(float));
    cudaMalloc((void**)&pii, d3 * sizeof(float));
    cudaMalloc((void**)&q, d3 * sizeof(float));
    cudaMalloc((void**)&qc, d3 * sizeof(float));
    cudaMalloc((void**)&qi, d3 * sizeof(float));
    cudaMalloc((void**)&qr, d3 * sizeof(float));
    cudaMalloc((void**)&qs, d3 * sizeof(float));
    cudaMalloc((void**)&den, d3 * sizeof(float));
    cudaMalloc((void**)&p, d3 * sizeof(float));
    cudaMalloc((void**)&delz, d3 * sizeof(float));
    cudaMalloc((void**)&rainncv, d2 * sizeof(float));
    cudaMalloc((void**)&snowncv, d2 * sizeof(float));
    cudaMalloc((void**)&sr, d2 * sizeof(float));
    cudaMalloc((void**)&rain, d2 * sizeof(float));
    cudaMalloc((void**)&snow, d2 * sizeof(float));

    cudaMemcpy(th, th, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pii, pii, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(q, q, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qc, qc, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qi, qi, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qr, qr, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qs, qs, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(den, den, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p, p, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delz, delz, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rainncv, rainncv, d2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(snowncv, snowncv, d2 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(teamX, teamY);
    dim3 block(BLOCK_X, BLOCK_Y);

    auto start = std::chrono::steady_clock::now();

    wsm<<<grid, block>>>(th, pii, q, qc, qi, qr, qs, den, p, delz,
        rain, rainncv,
        sr,
        snow, snowncv,
        delt,
        dips+1 , (ipe-ips+1) , 

        djps+1 , (jpe-jps+1) , 

        dkps+1 , (kpe-kps+1),  

        dips+1 , dipe ,        

        djps+1 , djpe ,        

        dkps+1 , dkpe ,        

        dips+1 , dipe ,        

        djps+1 , djpe ,        

        dkps+1 , dkpe ,        

        teamX , teamY );

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    cudaMemcpy(rain, rain, d2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snow, snow, d2 * sizeof(float), cudaMemcpyDeviceToHost);

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    FREE(th) ;
    FREE(pii) ;
    FREE(q) ;
    FREE(qc) ;
    FREE(qi) ;
    FREE(qr) ;
    FREE(qs) ;
    FREE(den) ;
    FREE(p) ;
    FREE(delz) ;
    FREE(rain) ;
    FREE(rainncv) ;
    FREE(sr) ;
    FREE(snow) ;
    FREE(snowncv) ;
  }

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);
  return(0) ;
}
```