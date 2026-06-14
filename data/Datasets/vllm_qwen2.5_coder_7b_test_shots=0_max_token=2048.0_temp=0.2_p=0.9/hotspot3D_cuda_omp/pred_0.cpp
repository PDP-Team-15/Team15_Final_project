Here is the translation of the `3D.cu` and `3D_helper.cu` files from CUDA to OpenMP:

### 3D_omp.c
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
    const float* __restrict__ tIn, 
    const float* __restrict__ pIn, 
          float* __restrict__ tOut,
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

  #pragma omp parallel for collapse(2)
  for (int j = 0; j < numRows; ++j) {
    for (int i = 0; i < numCols; ++i) {
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
  }
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
  float* tIn   = (float*) calloc(size,sizeof(float));
  float* pIn   = (float*) calloc(size,sizeof(float));
  float* tCopy = (float*) malloc(size * sizeof(float));
  float* tOut  = (float*) calloc(size,sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  memcpy(tCopy,tIn, size * sizeof(float));

  clock_t start = clock();

  for(int j = 0; j < iterations; j++)
  {
    hotspot3d(tIn, pIn, tOut, numCols, numRows, layers,
              ce, cw, cn, cs, ct, cb, cc, stepDivCap);

    float* temp = tIn;
    tIn = tOut;
    tOut = temp;
  }

  clock_t stop = clock();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tOut,answer,numRows*numCols*layers);
  float time = (float)(stop - start) / CLOCKS_PER_SEC;
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
```

### 3D_helper_omp.c
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STR_SIZE 256

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

 