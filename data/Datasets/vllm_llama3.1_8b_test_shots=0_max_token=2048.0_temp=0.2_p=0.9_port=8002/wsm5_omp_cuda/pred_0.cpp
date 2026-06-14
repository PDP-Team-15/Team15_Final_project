Here's the equivalent CUDA code for the given OpenMP code:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "utils.h"
#include "kernel.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void wsm_kernel(float *th, float *pii, float *q, float *qc, float *qi, float *qr, float *qs, float *den, float *p, float *delz,
                            float *rain, float *rainncv, float *sr, float *snow, float *snowncv,
                            float delt, int dips, int dipe, int djps, int djpe, int dkps, int dkpe,
                            int dips2, int dipe2, int djps2, int djpe2, int dkps2, int dkpe2,
                            int teamX, int teamY) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  int idy2 = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= dips2 && idx < dips2 + dipe2 && idy >= djps2 && idy < djps2 + djpe2 && idy2 >= dkps2 && idy2 < dkps2 + dkpe2) {
    int idx3 = idx - dips2;
    int idy3 = idy - djps2;
    int idy4 = idy2 - dkps2;

    float th_val = th[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float pii_val = pii[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float q_val = q[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float qc_val = qc[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float qi_val = qi[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float qr_val = qr[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float qs_val = qs[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float den_val = den[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float p_val = p[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float delz_val = delz[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float rain_val = rain[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float rainncv_val = rainncv[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float sr_val = sr[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float snow_val = snow[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];
    float snowncv_val = snowncv[idx3 + idy3 * (dipe2 - dips2) + idy4 * (dipe2 - dips2) * (djpe2 - djps2)];

    wsm(th_val, pii_val, q_val, qc_val, qi_val, qr_val, qs_val, den_val, p_val, delz_val,
        rain_val, rainncv_val, sr_val, snow_val, snowncv_val, delt, dips, dipe, djps, djpe, dkps, dkpe,
        dips, dipe, djps, djpe, dkps, dkpe, dips, dipe, djps, djpe, dkps, dkpe, teamX, teamY);
  }
}

int main(int argc, char* argv[]) {
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

    int d3_blocks = (d3 + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    int d2_blocks = (d2 + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;

    dim3 grid(d3_blocks, d2_blocks);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    wsm_kernel<<<grid, block>>>(th, pii, q, qc, qi, qr, qs, den, p, delz,
                                rain, rainncv, sr, snow, snowncv,
                                delt,
                                dips+1, dipe, djps+1, djpe, dkps+1, dkpe,
                                dips+1, dipe, djps+1, djpe, dkps+1, dkpe,
                                dips+1, dipe, djps+1, djpe, dkps+1, dkpe,
                                dips+1, dipe, djps+1, djpe, dkps+1, dkpe,
                                teamX, teamY);

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    cudaMemset(rain, 0, d2 * sizeof(float));
    cudaMemset(snow, 0, d2 * sizeof(float));

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    FREE(th) ;
    FREE(pii) ;
