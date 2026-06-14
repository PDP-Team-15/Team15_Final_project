```cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include "utils.h"
#include "kernel.h"

#define XXX 16
#define YYY 16

#define ALLOC3(var) var = (float*)malloc((59+1)*(45+1)*2*sizeof(float))
#define ALLOC2(var) var = (float*)malloc((45+1)*2*sizeof(float))
#define FREE(var) free(var)

__global__ void wsm_kernel(float *th, float *pii, float *q, float *qc, float *qi, float *qr, float *qs, float *den, float *p, float *delz, float *rain, float *rainncv, float *sr, float *snow, float *snowncv, float delt, int dips, int ime, int djps, int jpe, int dkps, int kpe, int dips2, int dipe, int djps2, int djpe, int dkps2, int dkpe, int teamX, int teamY) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= dips2 || idx >= dips2 + dipe || idy >= djps2 || idy >= djps2 + djpe) return;
  int k = dkps2 + (idx - dips2) * dkpe + (idy - djps2) * dkpe;
  int i = dips2 + (idx - dips2) * dipe;
  int j = djps2 + (idy - djps2) * djpe;
  int idx2 = idx * ime + j;
  int idy2 = idy * jpe;
  int idx3 = idx2 * kme + k;
  int idy3 = idy2 * kpe;
  int idx4 = idx3 * (jme - jms + 1) + j;
  int idy4 = idy3 * (ime - ims + 1) + i;
  int idx5 = idx4 * (kme - kms + 1) + k;
  int idy5 = idy4;
  th[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = pii[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = q[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = qc[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = qi[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = qr[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = qs[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = den[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = p[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)] = delz[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)];
  rain[idx2 + idy2 * (ime - ims + 1)] += rainncv[idx2 + idy2 * (ime - ims + 1)];
  snow[idx2 + idy2 * (ime - ims + 1)] += snowncv[idx2 + idy2 * (ime - ims + 1)];
  sr[idy2 * (ime - ims + 1) + idx2] += delt * den[idx5 + idy5 * (ime - ims + 1) * (jme - jms + 1)];
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  float *th, *pii, *q, *qc, *qi, *qr, *qs, *den, *p, *delz, *rain, *rainncv, *sr, *snow, *snowncv;
  float *d_th, *d_pii, *d_q, *d_qc, *d_qi, *d_qr, *d_qs, *d_den, *d_p, *d_delz, *d_rain, *d_rainncv, *d_sr, *d_snow, *d_snowncv;

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

    cudaMalloc((void**)&d_th, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_pii, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_q, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_qc, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_qi, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_qr, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_qs, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_den, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_p, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_delz, (59+1)*(45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_rain, (45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_rainncv, (45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_sr, (45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_snow, (45+1)*2*sizeof(float));
    cudaMalloc((void**)&d_snowncv, (45+1)*2*sizeof(float));

    cudaMemcpy(d_th, th, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pii, pii, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qc, qc, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qi, qi, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qr, qr, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qs, qs, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_den, den, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delz, delz, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rain, rain, (45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rainncv, rainncv, (45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sr, sr, (45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snow, snow, (45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snowncv, snowncv, (45+1)*2*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((ime-ims+1) / threadsPerBlock.x + 1, (jpe-jps+1) / threadsPerBlock.y + 1);
    wsm_kernel<<<numBlocks, threadsPerBlock>>>(d_th, d_pii, d_q, d_qc, d_qi, d_qr, d_qs, d_den, d_p, d_delz, d_rain, d_rainncv, d_sr, d_snow, d_snowncv, delt, dips+1, ime, djps+1, jpe, dkps+1, kpe, dips+1, dipe, djps+1, djpe, dkps+1, dkpe, teamX, teamY);
    cudaDeviceSynchronize();

    cudaMemcpy(th, d_th, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pii, d_pii, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(q, d_q, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(qc, d_qc, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qi, d_qi, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qr, d_qr, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qs, d_qs, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(den, d_den, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, d_p, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(delz, d_delz, (59+1)*(45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rain, d_rain, (45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rainncv, d_rainncv, (45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sr, d_sr, (45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snow, d_snow, (45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snowncv, d_snowncv, (45+1)*2*sizeof(float), cudaMemcpyDeviceToHost);

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
    FREE(snowncv);
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
    cudaFree(d_rain);
    cudaFree(d_rainncv);
    cudaFree(d_sr);
    cudaFree(d_snow);
    cudaFree(d_snowncv);
  }

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);
  return 0;
}
```