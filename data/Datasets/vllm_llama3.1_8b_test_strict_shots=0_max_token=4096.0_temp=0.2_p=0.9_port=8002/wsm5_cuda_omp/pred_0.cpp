```cpp
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

  int num_threads_x = XXX, num_threads_y = YYY;
  int num_threads_x_total = (ipe-ips+1) / num_threads_x + remx, num_threads_y_total = (jpe-jps+1) / num_threads_y + remy;

  float rain_sum = 0, snow_sum = 0;

  long time = 0;
  for (int i = 0; i < repeat; i++) {
    // Allocate memory
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

    // Kernel execution
    #pragma omp parallel for num_threads(num_threads_x_total) schedule(static)
    for (int k = kms; k <= kme; k++) {
      for (int j = jms; j <= jme; j++) {
        for (int i = ims; i <= ime; i++) {
          wsm(
            th_d, pii_d, q_d, qc_d, qi_d, qr_d, qs_d, den_d, p_d, delz_d,
            rain_d, rainncv_d,
            sr_d,
            snow_d, snowncv_d,
            delt,
            dips+1 , (ipe-ips+1) , 

            djps+1 , (jpe-jps+1) , 

            dkps+1 , (kpe-kps+1),  

            dips+1 , dipe ,        

            djps+1 , djpe ,        

            dkps+1 , dkpe,         

            dips+1 , dipe ,        

            djps+1 , djpe ,        

            dkps+1 , dkpe) ;        
        }
      }
    }

    // Synchronize threads
    #pragma omp barrier

    // Copy results
    FROMDEV2(rain) ;
    FROMDEV2(snow) ;

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    // Deallocate memory
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