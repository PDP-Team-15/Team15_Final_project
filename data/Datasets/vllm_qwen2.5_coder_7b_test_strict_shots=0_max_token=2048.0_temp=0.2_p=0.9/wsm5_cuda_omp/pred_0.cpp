```cpp
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

  #pragma omp parallel for collapse(3)
  for (int k = 0; k < (kme-kms+1); k++) {
    for (int j = 0; j < (jme-jms+1); j++) {
      for (int i = 0; i < (ipe-ips+1); i++) {
        // Kernel call here
        wsm(
          th[k*(d2)+j*(dip)+i], pii[k*(d2)+j*(dip)+i], q[k*(d2)+j*(dip)+i], qc[k*(d2)+j*(dip)+i], qi[k*(d2)+j*(dip)+i], qr[k*(d2)+j*(dip)+i], qs[k*(d2)+j*(dip)+i], den[k*(d2)+j*(dip)+i], p[k*(d2)+j*(dip)+i], delz[k*(d2)+j*(dip)+i],
          rain[k*(d2)+j*(dip)+i], rainncv[k*(d2)+j*(dip)+i],
          sr[k*(d2)+j*(dip)+i],
          snow[k*(d2)+j*(dip)+i], snowncv[k*(d2)+j*(dip)+i],
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
    }
  }

  long time = 0;
  for (int i = 0; i < repeat; i++) {
    
    TODEV3(pii) ;
    TODEV3(den) ;
    TODEV3(p) ;
    TODEV3(delz) ;

    TODEV3(th) ;
    TODEV3(q) ;
    TODEV3(qc) ;
    TODEV3(qi) ;
    TODEV3(qr) ;
    TODEV3(qs) ;
    TODEV2(rain) ;
    TODEV2(rainncv) ;
    TODEV2(sr) ;
    TODEV2(snow) ;
    TODEV2(snowncv) ;

    #pragma omp barrier
    auto start = std::chrono::steady_clock::now();

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < (kme-kms+1); k++) {
      for (int j = 0; j < (jme-jms+1); j++) {
        for (int i = 0; i < (ipe-ips+1); i++) {
          // Kernel call here
          wsm(
            th[k*(d2)+j*(dip)+i], pii[k*(d2)+j*(dip)+i], q[k*(d2)+j*(dip)+i], qc[k*(d2)+j*(dip)+i], qi[k*(d2)+j*(dip)+i], qr[k*(d2)+j*(dip)+i], qs[k*(d2)+j*(dip)+i], den[k*(d2)+j*(dip)+i], p[k*(d2)+j*(dip)+i], delz[k*(d2)+j*(dip)+i],
            rain[k*(d2)+j*(dip)+i], rainncv[k*(d2)+j*(dip)+i],
            sr[k*(d2)+j*(dip)+i],
            snow[k*(d2)+j*(dip)+i], snowncv[k*(d2)+j*(dip)+i],
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
      }
    }

    #pragma omp barrier
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    FROMDEV2(rain) ;
    FROMDEV2(snow) ;

    float rain_sum = 0, snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    FREE(pii) ;
    FREE(den) ;
    FREE(p) ;
    FREE(delz) ;
    FREE(th) ;
    FREE(q) ;
    FREE(qc) ;
    FREE(qi) ;
    FREE(qr) ;
    FREE(qs) ;
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