#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "utils.h"
#include "kernel.h"
#include <cuda_runtime.h>

#define ALLOC3(var) cudaMalloc(&var, d3 * sizeof(float))
#define ALLOC2(var) cudaMalloc(&var, d2 * sizeof(float))

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
    // Allocate host memory
    ALLOC3(th);
    ALLOC3(pii);
    ALLOC3(q);
    ALLOC3(qc);
    ALLOC3(qi);
    ALLOC3(qr);
    ALLOC3(qs);
    ALLOC3(den);
    ALLOC3(p);
    ALLOC3(delz);
    ALLOC2(rain);
    ALLOC2(rainncv);
    ALLOC2(sr);
    ALLOC2(snow);
    ALLOC2(snowncv);

    // Allocate device memory and copy data
    float *d_th, *d_pii, *d_q, *d_qc, *d_qi, *d_qr, *d_qs, *d_den, *d_p, *d_delz;
    float *d_rainncv, *d_snowncv, *d_sr, *d_rain, *d_snow;

    cudaMalloc(&d_th, d3 * sizeof(float));
    cudaMalloc(&d_pii, d3 * sizeof(float));
    cudaMalloc(&d_q, d3 * sizeof(float));
    cudaMalloc(&d_qc, d3 * sizeof(float));
    cudaMalloc(&d_qi, d3 * sizeof(float));
    cudaMalloc(&d_qr, d3 * sizeof(float));
    cudaMalloc(&d_qs, d3 * sizeof(float));
    cudaMalloc(&d_den, d3 * sizeof(float));
    cudaMalloc(&d_p, d3 * sizeof(float));
    cudaMalloc(&d_delz, d3 * sizeof(float));
    cudaMalloc(&d_rainncv, d2 * sizeof(float));
    cudaMalloc(&d_snowncv, d2 * sizeof(float));
    cudaMalloc(&d_sr, d2 * sizeof(float));
    cudaMalloc(&d_rain, d2 * sizeof(float));
    cudaMalloc(&d_snow, d2 * sizeof(float));

    cudaMemcpy(d_th, th, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pii, pii, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qc, qc, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qi, qi, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qr, qr, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qs, qs, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_den, den, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delz, delz, d3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rainncv, rainncv, d2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snowncv, snowncv, d2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sr, sr, d2 * sizeof(float), cudaMemcpyHostToDevice);

    int remx = (ipe-ips+1) % XXX != 0 ? 1 : 0;
    int remy = (jpe-jps+1) % YYY != 0 ? 1 : 0;

    const int teamX = (ipe-ips+1) / XXX + remx;
    const int teamY = (jpe-jps+1) / YYY + remy;

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((dipe + blockSize.x - 1) / blockSize.x,
                  (djpe + blockSize.y - 1) / blockSize.y, 1);

    auto start = std::chrono::steady_clock::now();
    wsm<<<gridSize, blockSize>>>(
        d_th, d_pii, d_q, d_qc, d_qi, d_qr, d_qs, d_den, d_p, d_delz,
        d_rain, d_rainncv, d_sr, d_snow, d_snowncv, delt,
        dips+1, dipe, djps+1, djpe, dkps+1, dkpe, teamX, teamY,
        ips, ipe, jps, jpe, kms, kme, blockSize.x, blockSize.y, blockSize.z);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Copy results back to host
    cudaMemcpy(rain, d_rain, d2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snow, d_snow, d2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_th);
    cudaFree(d_pii);
    cudaFree(d_q);
    cudaFree(d_qc);
    cudaFree(d_qi);
    cudaFree(d_qr);
    cudaFree(d_qs);
    cudaFree(d_den);
    cudaFree(d_p);
    cudaFree(d_delz);
    cudaFree(d_rainncv);
    cudaFree(d_snowncv);
    cudaFree(d_sr);
    cudaFree(d_rain);
    cudaFree(d_snow);

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    // Free host memory
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
  }

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);
  return(0);
}