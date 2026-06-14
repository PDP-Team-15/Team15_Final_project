```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

  int num_threads = XXX * YYY;
  int num_blocks_x = (ipe-ips+1) / XXX + remx;
  int num_blocks_y = (jpe-jps+1) / YYY + remy;

  float rain_sum = 0, snow_sum = 0;

  long time = 0;
  for (int i = 0; i < repeat; i++) {
    // Allocate memory on host
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

    // Initialize memory on host
    for (int k = 0; k < d3; k++) {
      th[k] = 0;
      pii[k] = 0;
      q[k] = 0;
      qc[k] = 0;
      qi[k] = 0;
      qr[k] = 0;
      qs[k] = 0;
      den[k] = 0;
      p[k] = 0;
      delz[k] = 0;
    }
    for (int i = 0; i < d2; i++) {
      rain[i] = 0;
      rainncv[i] = 0;
      sr[i] = 0;
      snow[i] = 0;
      snowncv[i] = 0;
    }

    // Copy memory from host to device
    #pragma omp target teams distribute parallel for map(to: th[0:d3], pii[0:d3], q[0:d3], qc[0:d3], qi[0:d3], qr[0:d3], qs[0:d3], den[0:d3], p[0:d3], delz[0:d3]) map(from: rain[0:d2], rainncv[0:d2], sr[0:d2], snow[0:d2], snowncv[0:d2]))
    {
      for (int k = 0; k < d3; k++) {
        th[k] = 0;
        pii[k] = 0;
        q[k] = 0;
        qc[k] = 0;
        qi[k] = 0;
        qr[k] = 0;
        qs[k] = 0;
        den[k] = 0;
        p[k] = 0;
        delz[k] = 0;
      }
      for (int i = 0; i < d2; i++) {
        rain[i] = 0;
        rainncv[i] = 0;
        sr[i] = 0;
        snow[i] = 0;
        snowncv[i] = 0;
      }
    }

    // Launch kernel
    #pragma omp target teams distribute parallel for map(to: th[0:d3], pii[0:d3], q[0:d3], qc[0:d3], qi[0:d3], qr[0:d3], qs[0:d3], den[0:d3], p[0:d3], delz[0:d3]) map(from: rain[0:d2], rainncv[0:d2], sr[0:d2], snow[0:d2], snowncv[0:d2]))
    {
      wsm(th, pii, q, qc, qi, qr, qs, den, p, delz,
          rain, rainncv,
          sr,
          snow, snowncv,
          delt,
          dips+1 , (ipe-ips+1) , 
          djps+1 , (jpe-jps+1) , 
          dkps+1 , (kpe-kps+1),  
          dips+1 , dipe ,        
          djps+1 , djpe ,        
          dkps+1 , dkpe,         
          dips+1 , dipe ,        
          djps+1 , djpe ,        
          dkps+1 , dkpe);
    }

    // Copy memory from device to host
    #pragma omp target teams distribute parallel for map(from: rain[0:d2], rainncv[0:d2], sr[0:d2], snow[0:d2], snowncv[0:d2]))
    {
      for (int i = 0; i < d2; i++) {
        rain[i] = 0;
        rainncv[i] = 0;
        sr[i] = 0;
        snow[i] = 0;
        snowncv[i] = 0;
      }
    }

    // Calculate sums
    float rain_sum_host = 0, snow_sum_host = 0;
    #pragma omp target teams distribute parallel for map(to: rain[0:d2], snow[0:d2])
    {
      for (int i = 0; i < d2; i++) {
        rain_sum_host += rain[i];
        snow_sum_host += snow[i];
      }
    }

    // Free memory
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
  printf("Checksum: rain = %f snow = %f\n", rain_sum_host, snow_sum_host);

  return(0) ;
}
```

Note that the `wsm` function is assumed to be defined in the `kernel.h` header file, and it is not shown here. Also, the `XXX` and `YYY` values are assumed to be defined elsewhere in the code.