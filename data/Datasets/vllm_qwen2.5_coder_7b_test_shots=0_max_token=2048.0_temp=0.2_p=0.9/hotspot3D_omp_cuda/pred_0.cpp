Here is the CUDA translation of the provided OpenMP code:

```cpp
// 3D.cu
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

#define FACTOR_CHIP  0.5

__global__ void kernel(float* tIn, float* pIn, float* tOut, int numCols, int numRows, int layers, float Cap, float Rx, float Ry, float Rz, float dt, float amb_temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int c = idx + idy * numCols + idz * numCols * numRows;
    int xy = numCols * numRows;

    int W = (idx == 0) ? c : c - 1;
    int E = (idx == numCols - 1) ? c : c + 1;
    int N = (idy == 0) ? c : c - numCols;
    int S = (idy == numRows - 1) ? c : c + numCols;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c + xy];
    tOut[c] = CC * temp2 + CW * tIn[W] + CE * tIn[E] + CS * tIn[S] + CN * tIn[N] + CB * temp1 + CT * temp3 + STEP_DIV_CAP * pIn[c] + CT * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < layers - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];
        tOut[c] = CC * temp2 + CW * tIn[W] + CE * tIn[E] + CS * tIn[S] + CN * tIn[N] + CB * temp1 + CT * temp3 + STEP_DIV_CAP * pIn[c] + CT * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = CC * temp2 + CW * tIn[W] + CE * tIn[E] + CS * tIn[S] + CN * tIn[N] + CB * temp1 + CT * temp3 + STEP_DIV_CAP * pIn[c] + CT * amb_temp;
}

void usage(int argc, char **argv) {
    std::cerr << "Usage: " << argv[0] << " <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n";
    std::cerr << "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n";
    std::cerr << "\t<layers>  - number of layers in the grid (positive integer)\n";
    std::cerr << "\t<iteration> - number of iterations\n";
    std::cerr << "\t<powerFile>  - name of the file containing the initial power values of each cell\n";
    std::cerr << "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n";
    std::cerr << "\t<outputFile - output file\n";
    exit(1);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        usage(argc, argv);
    }

    char *pfile, *tfile, *ofile;
    int iterations = std::stoi(argv[3]);

    pfile = argv[4];
    tfile = argv[5];
    ofile = argv[6];
    int numCols = std::stoi(argv[1]);
    int numRows = std::stoi(argv[1]);
    int layers = std::stoi(argv[2]);

    float dx = chip_height / numRows;
    float dy = chip_width / numCols;
    float dz = t_chip / layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;

    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;
    cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

    int size = numCols * numRows * layers;
    float* tIn = new float[size];
    float* pIn = new float[size];
    float* tCopy = new float[size];
    float* tOut = new float[size];
    float* sel;

    readinput(tIn, numRows, numCols, layers, tfile);
    readinput(pIn, numRows, numCols, layers, pfile);

    memcpy(tCopy, tIn, size * sizeof(float));

    long long start = get_time();

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y, (layers + blockSize.z - 1) / blockSize.z);

    for (int j = 0; j < iterations; j++) {
        kernel<<<gridSize, blockSize>>>(tIn, pIn, tOut, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp);
        cudaDeviceSynchronize();
    }

    long long stop = get_time();

    float* answer = new float[size];
    computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

    float acc = accuracy(sel, answer, numRows * numCols * layers);
    float time = static_cast<float>((stop - start) / (1000.0 * 1000.0));
    std::cout << "Device offloading time: " << time << " (s)\n";
    std::cout << "Root-mean-square error: " << acc << "\n";

    writeoutput(tOut, numRows, numCols, layers, ofile);

    delete[] answer;
    delete[] tIn;
    delete[] pIn;
    delete[] tCopy;
    delete[] tOut;
    return 0;
}

// 3D_helper.cpp remains unchanged
```

### Explanation:
1. **Kernel Function**: The `kernel` function is defined as a CUDA kernel that performs the same computation as the original OpenMP loop.
2. **Grid and Block Dimensions**: The grid and block dimensions are set based on the problem size.
3. **Memory Management**: Dynamic memory allocation is used instead of `calloc` and `malloc`.
4. **CUDA API Calls**: Functions like `cudaDeviceSynchronize()` are used to ensure proper synchronization between host and device.

This CUDA code should provide similar performance and functionality to the original OpenMP code.