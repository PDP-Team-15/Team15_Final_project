```cpp
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

float  distance_host ( int i, float latitude_1, float longitude_1,
                       float latitude_2, float longitude_2 )
{
  float  dist ;
  float  rad_latitude_1 ;
  float  rad_latitude_2 ;
  float  rad_longitude_1 ;
  float  rad_longitude_2 ;

  float  BAZ , C , C2A , CU1 , CU2 , CX , CY , CZ ,
         D , E , FAZ , SA , SU1 , SX  , SY , TU1 , TU2 , X , Y ; 

  const float GDC_DEG_TO_RAD = 3.141592654 / 180.0 ;  

  const float GDC_FLATTENING = 1.0 - ( 6356752.31424518 / 6378137.0 ) ; 
  const float GDC_ECCENTRICITY = ( 6356752.31424518 / 6378137.0 ) ; 
  const float GDC_ELLIPSOIDAL =  1.0 / ( 6356752.31414 / 6378137.0 ) / ( 6356752.31414 / 6378137.0 ) - 1.0 ;
  const float GDC_SEMI_MINOR = 6356752.31424518f;
  const float EPS = 0.5e-5f;

  rad_longitude_1 = longitude_1 * GDC_DEG_TO_RAD ;
  rad_latitude_1 = latitude_1 * GDC_DEG_TO_RAD ;
  rad_longitude_2 = longitude_2 * GDC_DEG_TO_RAD ;
  rad_latitude_2 = latitude_2 * GDC_DEG_TO_RAD ;

  TU1 = GDC_ECCENTRICITY * sinf ( rad_latitude_1 ) /
    cosf ( rad_latitude_1 ) ;
  TU2 = GDC_ECCENTRICITY * sinf ( rad_latitude_2 ) /
    cosf ( rad_latitude_2 ) ;

  CU1 = 1.0f / sqrtf ( TU1 * TU1 + 1.0f ) ;
  SU1 = CU1 * TU1 ;
  CU2 = 1.0f / sqrtf ( TU2 * TU2 + 1.0f ) ;
  dist = CU1 * CU2 ;
  BAZ = dist * TU2 ;
  FAZ = BAZ * TU1 ;
  X = rad_longitude_2 - rad_longitude_1 ;

  do {
    SX = sinf ( X ) ;
    CX = cosf ( X ) ;
    TU1 = CU2 * SX ;
    TU2 = BAZ - SU1 * CU2 * CX ;
    SY = sqrtf ( TU1 * TU1 + TU2 * TU2 ) ;
    CY = dist * CX + FAZ ;
    Y = atan2f ( SY, CY ) ;
    SA = dist * SX / SY ;
    C2A = - SA * SA + 1.0f;
    CZ = FAZ + FAZ ;
    if ( C2A > 0.0f ) CZ = -CZ / C2A + CY ;
    E = CZ * CZ * 2.0f - 1.0f ;
    C = ( ( -3.0f * C2A + 4.0f ) * GDC_FLATTENING + 4.0f ) * C2A *
      GDC_FLATTENING / 16.0f ;
    D = X ;
    X = ( ( E * CY * C + CZ ) * SY * C + Y ) * SA ;
    X = ( 1.0f - C ) * X * GDC_FLATTENING + rad_longitude_2 - rad_longitude_1 ;
  } while ( fabsf ( D - X ) > EPS );

  X = sqrtf ( GDC_ELLIPSOIDAL * C2A + 1.0f ) + 1.0f ;
  X = ( X - 2.0f ) / X ;
  C = 1.0f - X ;
  C = ( X * X / 4.0f + 1.0f ) / C ;
  D = ( 0.375f * X * X - 1.0f ) * X ;
  X = E * CY ;
  dist = 1.0f - E - E ;
  dist = ( ( ( ( SY * SY * 4.0f - 3.0f ) * dist * CZ * D / 6.0f -
          X ) * D / 4.0f + CZ ) * SY * D + Y ) * C * GDC_SEMI_MINOR ;
  return dist;
}

__global__ void distance_kernel(const float4* VA, float* VC, const int N, const int iteration) {

  __shared__ float GDC_DEG_TO_RAD;
  __shared__ float GDC_FLATTENING;
  __shared__ float GDC_ECCENTRICITY;
  __shared__ float GDC_ELLIPSOIDAL;
  __shared__ float GDC_SEMI_MINOR;
  __shared__ float EPS;

  __shared__ float TU1;
  __shared__ float TU2;
  __shared__ float CU1;
  __shared__ float CU2;
  __shared__ float BAZ;
  __shared__ float FAZ;
  __shared__ float X;
  __shared__ float SX;
  __shared__ float CX;
  __shared__ float SY;
  __shared__ float CY;
  __shared__ float Y;
  __shared__ float SA;
  __shared__ float C2A;
  __shared__ float CZ;
  __shared__ float D;
  __shared__ float E;

  __shared__ float rad_latitude_1;
  __shared__ float rad_longitude_1;
  __shared__ float rad_latitude_2;
  __shared__ float rad_longitude_2;

  int wiID = blockIdx.x * blockDim.x + threadIdx.x;

  if (wiID < N) {
    GDC_DEG_TO_RAD = 3.141592654 / 180.0;
    GDC_FLATTENING = 1.0 - ( 6356752.31424518 / 6378137.0 );
    GDC_ECCENTRICITY = ( 6356752.31424518 / 6378137.0 );
    GDC_ELLIPSOIDAL =  1.0 / ( 6356752.31414 / 6378137.0 ) / ( 6356752.31414 / 6378137.0 ) - 1.0;
    GDC_SEMI_MINOR = 6356752.31424518;
    EPS = 0.5e-5;

    rad_latitude_1  = VA[wiID].x * GDC_DEG_TO_RAD;
    rad_longitude_1 = VA[wiID].y * GDC_DEG_TO_RAD;
    rad_latitude_2  = VA[wiID].z * GDC_DEG_TO_RAD;
    rad_longitude_2 = VA[wiID].w * GDC_DEG_TO_RAD;

    TU1 = GDC_ECCENTRICITY * sinf ( rad_latitude_1 ) /
      cosf ( rad_latitude_1 );
    TU2 = GDC_ECCENTRICITY * sinf ( rad_latitude_2 ) /
      cosf ( rad_latitude_2 );

    CU1 = 1.0f / sqrtf ( TU1 * TU1 + 1.0f );
    CU2 = 1.0f / sqrtf ( TU2 * TU2 + 1.0f );
    dist = CU1 * CU2;
    BAZ = dist * TU2;
    FAZ = BAZ * TU1;
    X = rad_longitude_2 - rad_longitude_1;

    do {
      SX = sinf ( X );
      CX = cosf ( X );
      TU1 = CU2 * SX;
      TU2 = BAZ - SU1 * CU2 * CX;
      SY = sqrtf ( TU1 * TU1 + TU2 * TU2 );
      CY = dist * CX + FAZ;
      Y = atan2f ( SY, CY );
      SA = dist * SX / SY;
      C2A = - SA * SA + 1.0f;
      CZ = FAZ + FAZ;
      if ( C2A > 0.0f ) CZ = -CZ / C2A + CY;
      E = CZ * CZ * 2.0f - 1.0f;
      C = ( ( -3.0f * C2A + 4.0f ) * GDC_FLATTENING + 4.0f ) * C2A *
        GDC_FLATTENING / 16.0f;
      D = X;
      X = ( ( E * CY * C + CZ ) * SY * C + Y ) * SA;
      X = ( 1.0f - C ) * X * GDC_FLATTENING + rad_longitude_2 - rad_longitude_1;
    } while ( fabsf ( D - X ) > EPS );

    X = sqrtf ( GDC