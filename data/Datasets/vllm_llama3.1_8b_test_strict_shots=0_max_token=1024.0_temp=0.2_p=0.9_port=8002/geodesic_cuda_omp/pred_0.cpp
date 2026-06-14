```cpp
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <omp.h>

float distance_host(int i, float latitude_1, float longitude_1,
                     float latitude_2, float longitude_2)
{
  float dist;
  float rad_latitude_1;
  float rad_latitude_2;
  float rad_longitude_1;
  float rad_longitude_2;

  float BAZ, C, C2A, CU1, CU2, CX, CY, CZ, D, E, FAZ, SA, SU1, SX, SY, TU1, TU2, X, Y;

  const float GDC_DEG_TO_RAD = 3.141592654 / 180.0;

  const float GDC_FLATTENING = 1.0 - (6356752.31424518 / 6378137.0);
  const float GDC_ECCENTRICITY = (6356752.31424518 / 6378137.0);
  const float GDC_ELLIPSOIDAL = 1.0 / (6356752.31414 / 6378137.0) / (6356752.31414 / 6378137.0) - 1.0;
  const float GC_SEMI_MINOR = 6356752.31424518f;
  const float EPS = 0.5e-5f;

  rad_longitude_1 = longitude_1 * GDC_DEG_TO_RAD;
  rad_latitude_1 = latitude_1 * GDC_DEG_TO_RAD;
  rad_longitude_2 = longitude_2 * GDC_DEG_TO_RAD;
  rad_latitude_2 = latitude_2 * GDC_DEG_TO_RAD;

  TU1 = GDC_ECCENTRICITY * sinf(rad_latitude_1) / cosf(rad_latitude_1);
  TU2 = GDC_ECCENTRICITY * sinf(rad_latitude_2) / cosf(rad_latitude_2);

  CU1 = 1.0f / sqrtf(TU1 * TU1 + 1.0f);
  SU1 = CU1 * TU1;
  CU2 = 1.0f / sqrtf(TU2 * TU2 + 1.0f);
  dist = CU1 * CU2;
  BAZ = dist * TU2;
  FAZ = BAZ * TU1;
  X = rad_longitude_2 - rad_longitude_1;

  do {
    SX = sinf(X);
    CX = cosf(X);
    TU1 = CU2 * SX;
    TU2 = BAZ - SU1 * CU2 * CX;
    SY = sqrtf(TU1 * TU1 + TU2 * TU2);
    CY = dist * CX + FAZ;
    Y = atan2f(SY, CY);
    SA = dist * SX / SY;
    C2A = -SA * SA + 1.0f;
    CZ = FAZ + FAZ;
    if (C2A > 0.0f)
      CZ = -CZ / C2A + CY;
    E = CZ * CZ * 2.0f - 1.0f;
    C = ((-3.0f * C2A + 4.0f) * GDC_FLATTENING + 4.0f) * C2A * GDC_FLATTENING / 16.0f;
    D = X;
    X = ((E * CY * C + CZ) * SY * C + Y) * SA;
    X = (1.0f - C) * X * GDC_FLATTENING + rad_longitude_2 - rad_longitude_1;
  } while (fabsf(D - X) > EPS);

  X = sqrtf(GDC_ELLIPSOIDAL * C2A + 1.0f) + 1.0f;
  X = (X - 2.0f) / X;
  C = 1.0f - X;
  C = (X * X / 4.0f + 1.0f) / C;
  D = (0.375f * X * X - 1.0f) * X;
  X = E * CY;
  dist = 1.0f - E - E;
  dist = (((((SY * SY * 4.0f - 3.0f) * dist * CZ * D / 6.0f - X) * D / 4.0f + CZ) * SY * D + Y) * C * GC_SEMI_MINOR;
  return dist;
}

void kernel_distance(const float4* __restrict__ d_A, float* __restrict__ d_C, const int N)
{
  #pragma omp parallel