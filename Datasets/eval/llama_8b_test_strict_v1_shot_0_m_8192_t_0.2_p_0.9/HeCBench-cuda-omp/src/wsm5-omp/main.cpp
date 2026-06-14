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
#include "utils.h"
#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  float *th, *pii, *q;
  float *qc, *qi, *qr, *qs;
  float *den, *p, *delz;
  float *rain, *rainncv;
  float *sr;
  float *snow, *snowncv;

  float delt = 10.f;
  int ims = 0, ime = 59, jms = 0, jme = 45, kms = 0, kme = 2;
  int ips = 0, ipe = 59, jps = 0, jpe = 45, kps = 0, kpe = 2;
  int d3 = (ime - ims + 1) * (jme - jms + 1) * (kme - kms + 1);
  int d2 = (ime - ims + 1) * (jme - jms + 1);

  int dips = 0;
  int dipe = (ipe - ips + 1);
  int djps = 0;
  int djpe = (jpe - jps + 1);
  int dkps = 0;
  int dkpe = (kpe - kps + 1);

  int remx = (ipe - ips + 1) % XXX != 0 ? 1 : 0;
  int remy = (jpe - jps + 1) % YYY != 0 ? 1 : 0;

  int num_threads = XXX;
  int num_blocks_x = (ipe - ips + 1) / XXX + remx;
  int num_blocks_y = (jpe - jps + 1) / YYY + remy;

  float rain_sum = 0, snow_sum = 0;

  long time = 0;
  for (int i = 0; i < repeat; i++) {
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

    TODEV3(pii);
    TODEV3(den);
    TODEV3(p);
    TODEV3(delz);

    TODEV3(th);
    TODEV3(q);
    TODEV3(qc);
    TODEV3(qi);
    TODEV3(qr);
    TODEV3(qs);
    TODEV2(rain);
    TODEV2(rainncv);
    TODEV2(sr);
    TODEV2(snow);
    TODEV2(snowncv);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int k = kms; k <= kme; k++) {
      for (int j = jms; j <= jme; j++) {
        for (int i = ims; i <= ime; i++) {
          // Kernel implementation
          wsm(
            th[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            pii[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            q[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            qc[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            qi[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            qr[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            qs[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            den[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            p[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            delz[i + j * (ime - ims + 1) + k * (jme - jms + 1) * (ime - ims + 1)],
            rain[i + j * (ime - ims + 1)],
            rainncv[i + j * (ime - ims + 1)],
            sr[i + j * (ime - ims + 1)],
            snow[i + j * (ime - ims + 1)],
            snowncv[i + j * (ime - ims + 1)],
            delt,
            dips + 1, (ipe - ips + 1),
            djps + 1, (jpe - jps + 1),
            dkps + 1, (kpe - kps + 1),
            dips + 1, dipe,
            djps + 1, djpe,
            dkps + 1, dkpe,
            dips + 1, dipe,
            djps + 1, djpe,
            dkps + 1, dkpe
          );
        }
      }
    }

    FROMDEV2(rain);
    FROMDEV2(snow);

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    FREE(pii);
    FREE(den);
    FREE(p);
    FREE(delz);
    FREE(th);
    FREE(q);
    FREE(qc);
    FREE(qi);
    FREE(qr);
    FREE(qs);
    FREE(rain);
    FREE(rainncv);
    FREE(sr);
    FREE(snow);
    FREE(snowncv);
  }

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);

  return 0;
}