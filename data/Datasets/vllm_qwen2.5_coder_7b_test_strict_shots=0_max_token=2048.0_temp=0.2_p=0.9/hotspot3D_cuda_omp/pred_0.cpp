```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

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
  std::cerr << "Usage: " << argv[0] << " <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>" << std::endl;
  std::cerr << "\t<rows/cols>  - number of rows/cols in the grid (positive integer)" << std::endl;
  std::cerr << "\t<layers>  - number of layers in the grid (positive integer)" << std::endl;
  std::cerr << "\t<iteration> - number of iterations" << std::endl;
  std::cerr << "\t<powerFile>  - name of the file containing the initial power values of each cell" << std::endl;
  std::cerr << "\t<tempFile>  - name of the file containing the initial temperature values of each cell" << std::endl;
  std::cerr << "\t<outputFile - output file" << std::endl;
  exit(1);
}

void hotspot3d(
    const std::vector<float>& tIn, 
    const std::vector<float>& pIn, 
    std::vector<float>& tOut,
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
  int iterations = std::atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = std::atoi(argv[1]);
  int numRows      = std::atoi(argv[1]);
  int layers       = std::atoi(argv[2]);

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
  std::vector<float> tIn(size, 0.0);
  std::vector<float> pIn(size, 0.0);
  std::vector<float> tCopy(size, 0.0);
  std::vector<float> tOut(size, 0.0);

  readinput(tIn, numRows, numCols, layers, tfile);
  readinput(pIn, numRows, numCols, layers, pfile);

  std::copy(tIn.begin(), tIn.end(), tCopy.begin());

  double start = omp_get_wtime();

  for(int j = 0; j < iterations; j++)
  {
    hotspot3d(tIn, pIn, tOut, numCols, numRows, layers,
              ce, cw, cn, cs, ct, cb, cc, stepDivCap);

    std::swap(tIn, tOut);
  }

  double stop = omp_get_wtime();

  std::vector<float> answer(size, 0.0);
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tIn.data(), answer.data(), size);
  float time = static_cast<float>((stop - start) / (1000.0 * 1000.0));
  std::cout << "Device offloading time: " << time << " (s)" << std::endl;
  std::cout << "Root-mean-square error: " << acc << std::endl;

  writeoutput(tIn.data(), numRows, numCols, layers, ofile);

  return 0;
}

void readinput(std::vector<float>& vect, int grid_rows, int grid_cols, int layers, char *file) {
  int i,j,k;
  std::ifstream fp(file);
  if (!fp.is_open()) {
    std::cerr << "The file was not opened" << std::endl;
    exit(1);
  }

  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
      for (k=0; k < layers; k++)
      {
        float val;
        fp >> val;
        vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
      }
  fp.close();  
}

void writeoutput(const float *vect, int grid_rows, int grid_cols, int layers, char *file) {
  int i,j,k, index=0;
  std::ofstream fp(file);
  if (!fp.is_open()) {
    std::cerr << "The file was not opened" << std::endl;
    return;
  }

  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
      for (k=0; k < layers; k++)
      {
        fp << index << "\t" << vect[i*grid_cols+j+k*grid_rows*grid_cols] << "\n";
        index++;
      }

  fp.close();  
}

void computeTempCPU(const float *pIn, const float* tIn, float *tOut, 
    int nx, int ny, int nz, float Cap, 
    float Rx, float Ry, float Rz, 
    float dt, float amb_temp, int numiter) 
{   
  float ce