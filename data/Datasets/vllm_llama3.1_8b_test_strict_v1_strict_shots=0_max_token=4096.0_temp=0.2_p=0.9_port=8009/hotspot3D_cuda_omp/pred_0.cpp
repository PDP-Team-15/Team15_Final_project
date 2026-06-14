Here is the translated code from CUDA to OpenMP:

```c
#include <sys/types.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

#define FACTOR_CHIP  0.5

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}

void hotspot3d(
    const float*__restrict__ tIn, 
    const float*__restrict__ pIn, 
          float*__restrict__ tOut,
    const int numCols, 
    const int numRows, 
    const int layers,
    const float ce, 
    const float cw,
    const float cn, 
    const float cs,
    const float ct,
    const float cb,
    const float cc,
    const float stepDivCap)
{
  float amb_temp = 80.0;

  int i = omp_get_thread_num();
  int j = omp_get_thread_num() + omp_get_num_threads();
  int c = i + j * numCols;
  int xy = numCols * numRows;

  int W = (i == 0)        ? c : c - 1;
  int E = (i == numCols-1)     ? c : c + 1;
  int N = (j == 0)        ? c : c - numCols;
  int S = (j == numRows-1)     ? c : c + numCols;

  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c+xy];
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
  c += xy;
  W += xy;
  E += xy;
  N += xy;
  S += xy;

  #pragma omp parallel for
  for (int k = 1; k < layers-1; ++k) {
    temp1 = temp2;
    temp2 = temp3;
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
      + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
  }
  temp1 = temp2;
  temp2 = temp3;
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
}

int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc,argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw = stepDivCap/ Rx;
  cn               = cs = stepDivCap/ Ry;
  ct               = cb = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int size = numCols * numRows * layers;
  float* tIn   = (float*) malloc(size*sizeof(float));
  float* pIn   = (float*) malloc(size*sizeof(float));
  float* tCopy = (float*) malloc(size * sizeof(float));
  float* tOut  = (float*) malloc(size*sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  memcpy(tCopy,tIn, size * sizeof(float));

  long long start = get_time();

  #pragma omp parallel for
  for(int j = 0; j < iterations; j++)
  {
    hotspot3d(tIn, pIn, tOut, numCols, numRows, layers,
        ce, cw, cn, cs, ct, cb, cc, stepDivCap);
    float* temp = tIn;
    tIn = tOut;
    tOut = temp;
  }

  long long stop = get_time();

  float* answer = (float*)malloc(size * sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Device offloading time: %.3f (s)\n",time);
  printf("Root-mean-square error: %e\n",acc);

  writeoutput(tOut,numRows,numCols,layers,ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  return 0;
}

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void fatal(const char *s)
{
  fprintf(stderr, "Error: %s\n", s);
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {

  int i,j,k;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 )
    fatal( "The file was not opened" );

  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
      for (k=0; k <= layers-1; k++)
      {
        if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
        if (feof(fp))
          fatal("not enough lines in file");
        

        if ((sscanf(str, "%f", &val) != 1))
          fatal("invalid file format");
        vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
      }
  fclose(fp);  
}

void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {
  int i,j,k, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
    printf( "The file was not opened\n" );

  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
      for (k=0; k < layers; k++)
      {
        sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j+k*grid_rows*grid_cols]);
        fputs(str,fp);
        index++;
      }

  fclose(fp);  
}

void computeTempCPU(float *pIn, float* tIn, float *tOut, 
    int nx, int ny, int nz, float Cap, 
    float Rx, float Ry, float Rz, 
    float dt, float amb_temp, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce = cw =stepDivCap/ Rx;
  cn = cs =stepDivCap/ Ry;
  ct = cb =stepDivCap/ Rz;

  cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int c,w,e,n,s,b,t;
  int x,y,z;
  int i = 0;
  do{
    for(z = 0; z < nz; z++)
      for(y = 0; y < ny; y++)
        for(x = 0; x < nx; x++)
        {
          c = x + y * nx + z * nx * ny;

          w = (x == 0) ? c : c - 1;
          e = (x == nx - 1) ? c : c + 1;
          n = (y == 0) ? c : c - nx;
          s = (y == ny - 1) ? c : c + nx;
          b = (z == 0) ? c : c - nx * ny;
          t = (z == nz - 1) ? c : c + nx * ny;


          tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw +
                    tIn[t]*ct + tIn[b]*cb + (dt/Cap) * pIn[c] + ct*amb_temp;
        }
    float *temp = tIn;
    tIn = tOut;
    tOut = temp; 
    i++;
  }
  while(i < numiter);
}

float accuracy(float *arr1, float *arr2, int len)
{
  float err = 0.0; 
  int i;
  for(i = 0; i < len; i++)
  {
    err += (arr1[i]-arr2[i]) * (arr1[i]-arr2[i]);
  }

  return (float)sqrt(err/len);
}
```

Note that I've used OpenMP's `#pragma omp parallel for` directive to parallelize the loops. I've also replaced the CUDA memory management functions with standard C memory management functions. Additionally, I've removed the `cudaDeviceSynchronize()` calls, as they are not necessary in OpenMP.