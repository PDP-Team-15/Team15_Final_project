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
#include <omp.h>
#include "utils.h"
#include "kernel.h"

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

  int remx = (ipe-ips+1) % XXX != 0 ? 1 : 0 ;
  int remy = (jpe-jps+1) % YYY != 0 ? 1 : 0 ;

  th = (float*)malloc(d3 * sizeof(float));
  pii = (float*)malloc(d3 * sizeof(float));
  q = (float*)malloc(d3 * sizeof(float));
  qc = (float*)malloc(d3 * sizeof(float));
  qi = (float*)malloc(d3 * sizeof(float));
  qr = (float*)malloc(d3 * sizeof(float));
  qs = (float*)malloc(d3 * sizeof(float));
  den = (float*)malloc(d3 * sizeof(float));
  p = (float*)malloc(d3 * sizeof(float));
  delz = (float*)malloc(d3 * sizeof(float));
  rain = (float*)malloc(d2 * sizeof(float));
  rainncv = (float*)malloc(d2 * sizeof(float));
  sr = (float*)malloc(d2 * sizeof(float));
  snow = (float*)malloc(d2 * sizeof(float));
  snowncv = (float*)malloc(d2 * sizeof(float));

  float rain_sum = 0, snow_sum = 0;

  long time = 0;
  for (int i = 0; i < repeat; i++) {

    #pragma omp parallel for collapse(3)
    for (int k = dkps+1; k <= dkpe; k++) {
      for (int j = djps+1; j <= djpe; j++) {
        for (int i = dips+1; i <= dipe; i++) {
          wsm(th, pii, q, qc, qi, qr, qs, den, p, delz, rain, rainncv, sr, snow, snowncv, delt, i, j, k);
        }
      }
    }

    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel for collapse(3)
    for (int k = dkps+1; k <= dkpe; k++) {
      for (int j = djps+1; j <= djpe; j++) {
        for (int i = dips+1; i <= dipe; i++) {
          wsm(th, pii, q, qc, qi, qr, qs, den, p, delz, rain, rainncv, sr, snow, snowncv, delt, i, j, k);
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    free(th);
    free(pii);
    free(q);
    free(qc);
    free(qi);
    free(qr);
    free(qs);
    free(den);
    free(p);
    free(delz);
    free(rain);
    free(rainncv);
    free(sr);
    free(snow);
    free(snowncv);
  }

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);

  return(0) ;
}