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
#include <omp.h>
#include <chrono>

#define MAX_REPEAT 100

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

  int remx = (ipe-ips+1) % 32 != 0 ? 1 : 0 ;
  int remy = (jpe-jps+1) % 32 != 0 ? 1 : 0 ;

  int num_threads_x = 32;
  int num_threads_y = 32;
  int num_threads = num_threads_x * num_threads_y;

  float rain_sum = 0, snow_sum = 0;

  long time = 0;
  for (int i = 0; i < repeat; i++) {
    // Allocate memory
    pii = (float*)malloc(d3 * sizeof(float));
    den = (float*)malloc(d3 * sizeof(float));
    p = (float*)malloc(d3 * sizeof(float));
    delz = (float*)malloc(d3 * sizeof(float));
    th = (float*)malloc(d3 * sizeof(float));
    q = (float*)malloc(d3 * sizeof(float));
    qc = (float*)malloc(d3 * sizeof(float));
    qi = (float*)malloc(d3 * sizeof(float));
    qr = (float*)malloc(d3 * sizeof(float));
    qs = (float*)malloc(d3 * sizeof(float));
    rain = (float*)malloc(d2 * sizeof(float));
    rainncv = (float*)malloc(d2 * sizeof(float));
    sr = (float*)malloc(d2 * sizeof(float));
    snow = (float*)malloc(d2 * sizeof(float));
    snowncv = (float*)malloc(d2 * sizeof(float));

    // Initialize data
    for (int k = 0; k < d3; k++) {
      pii[k] = 0;
      den[k] = 0;
      p[k] = 0;
      delz[k] = 0;
      th[k] = 0;
      q[k] = 0;
      qc[k] = 0;
      qi[k] = 0;
      qr[k] = 0;
      qs[k] = 0;
    }
    for (int i = 0; i < d2; i++) {
      rain[i] = 0;
      rainncv[i] = 0;
      sr[i] = 0;
      snow[i] = 0;
      snowncv[i] = 0;
    }

    // Parallelize the kernel
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int k = 0; k < kme-kms+1; k++) {
      for (int j = 0; j < jme-jms+1; j++) {
        for (int i = 0; i < ime-ims+1; i++) {
          // Kernel code
          int idx = i + ims + (j + jms) * (ime - ims + 1) + (k + kms) * (jme - jms + 1) * (ime - ims + 1);
          int idx2 = i + ips + (j + jps) * (ipe - ips + 1);
          int idx3 = i + dips + (j + djps) * (dipe - dips);
          int idx4 = i + dkps + (k + dkps) * (dkpe - dkps);

          pii[idx] = 0;
          den[idx] = 0;
          p[idx] = 0;
          delz[idx] = 0;
          th[idx] = 0;
          q[idx] = 0;
          qc[idx] = 0;
          qi[idx] = 0;
          qr[idx] = 0;
          qs[idx] = 0;

          rain[idx2] = 0;
          rainncv[idx2] = 0;
          sr[idx2] = 0;
          snow[idx2] = 0;
          snowncv[idx2] = 0;

          pii[idx3] = 0;
          den[idx3] = 0;
          p[idx3] = 0;
          delz[idx3] = 0;
          th[idx3] = 0;
          q[idx3] = 0;
          qc[idx3] = 0;
          qi[idx3] = 0;
          qr[idx3] = 0;
          qs[idx3] = 0;

          rain[idx4] = 0;
          rainncv[idx4] = 0;
          sr[idx4] = 0;
          snow[idx4] = 0;
          snowncv[idx4] = 0;
        }
      }
    }

    // Measure execution time
    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int k = 0; k < kme-kms+1; k++) {
      for (int j = 0; j < jme-jms+1; j++) {
        for (int i = 0; i < ime-ims+1; i++) {
          // Kernel code
          int idx = i + ims + (j + jms) * (ime - ims + 1) + (k + kms) * (jme - jms + 1) * (ime - ims + 1);
          int idx2 = i + ips + (j + jps) * (ipe - ips + 1);
          int idx3 = i + dips + (j + djps) * (dipe - dips);
          int idx4 = i + dkps + (k + dkps) * (dkpe - dkps);

          pii[idx] = 0;
          den[idx] = 0;
          p[idx] = 0;
          delz[idx] = 0;
          th[idx] = 0;
          q[idx] = 0;
          qc[idx] = 0;
          qi[idx] = 0;
          qr[idx] = 0;
          qs[idx] = 0;

          rain[idx2] = 0;
          rainncv[idx2] = 0;
          sr[idx2] = 0;
          snow[idx2] = 0;
          snowncv[idx2] = 0;

          pii[idx3] = 0;
          den[idx3] = 0;
          p[idx3] = 0;
          delz[idx3] = 0;
          th[idx3] = 0;
          q[idx3] = 0;
          qc[idx3] = 0;
          qi[idx3] = 0;
          qr[idx3] = 0;
          qs[idx3] = 0;

          rain[idx4] = 0;
          rainncv[idx4] = 0;
          sr[idx4] = 0;
          snow[idx4] = 0;
          snowncv[idx4] = 0;
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Free memory
    free(pii);
    free(den);
    free(p);
    free(delz);
    free(th);
    free(q);
    free(qc);
    free(qi);
    free(qr);
    free(qs);
    free(rain);
    free(rainncv);
    free(sr);
    free(snow);
    free(snowncv);

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }
  }

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);

  return 0;
}