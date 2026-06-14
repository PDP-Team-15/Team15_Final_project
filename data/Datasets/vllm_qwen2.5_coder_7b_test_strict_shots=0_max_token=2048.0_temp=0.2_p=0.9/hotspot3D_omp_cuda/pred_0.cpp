```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

#define FACTOR_CHIP  0.5

__global__ void computeTemperatureKernel(float* tIn, float* pIn, float* tOut, int size, float Cap, float Rx, float Ry, float Rz, float dt, float amb_temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int numCols = sqrt(size / (chip_height / chip_width));
    int numRows = numCols;
    int layers = size / (numCols * numRows);

    float dx = chip_height / numRows;
    float dy = chip_width / numCols;
    float dz = t_chip / layers;

    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;
    cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

    int c, w, e, n, s, b, t;
    int xy = numCols * numRows;

    c = idx;
    w = (c % numCols == 0) ? c : c - 1;
    e = (c % numCols == numCols - 1) ? c : c + 1;
    n = (c < numCols) ? c : c - numCols;
    s = (c >= (numRows - 1) * numCols) ? c : c + numCols;
    b = (c < layers * numCols * numRows) ? c : c - layers * numCols * numRows;
    t = (c >= (layers - 1) * numCols * numRows) ? c : c + layers * numCols * numRows;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c + xy];
    tOut[c] = cc * temp2 + cw * tIn[w] + ce * tIn[e] + cs * tIn[s] + cn * tIn[n] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;

    for (int k = 1; k < layers - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];
        tOut[c] = cc * temp2 + cw * tIn[w] + ce * tIn[e] + cs * tIn[s] + cn * tIn[n] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
        c += xy;
        w += xy;
        e += xy;
        n += xy;
        s += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[w] + ce * tIn[e] + cs * tIn[s] + cn * tIn[n] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
}

void usage(int argc, char **argv) {
    std::cerr << "Usage: " << argv[0] << " <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>" << std::endl;
    std::cerr << "\t<rows/cols>  - number of rows/cols in the grid (positive integer)" << std::endl;
    std::cerr << "\t<layers>  - number of layers in the grid (positive integer)" << std::endl;
    std::cerr << "\t<iteration> - number of iterations" << std::endl;
    std::cerr << "\t<powerFile>  - name of the file containing the initial power values of each cell" << std::endl;
    std::cerr << "\t<tempFile>  - name of the file containing the initial temperature values of each cell" << std::endl;
    std::cerr << "\t<outputFile - output file" << std::endl;
    exit(1);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        usage(argc, argv);
    }

    char *pfile, *tfile, *ofile;
    int iterations = std::atoi(argv[3]);

    pfile = argv[4];
    tfile = argv[5];
    ofile = argv[6];
    int numCols = std::atoi(argv[1]);
    int numRows = std::atoi(argv[1]);
    int layers = std::atoi(argv[2]);

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
    std::vector<float> tIn(size);
    std::vector<float> pIn(size);
    std::vector<float> tCopy(size);
    std::vector<float> tOut(size);
    float* sel;

    readinput(tIn.data(), numRows, numCols, layers, tfile);
    readinput(pIn.data(), numRows, numCols, layers, pfile);

    std::copy(tIn.begin(), tIn.end(), tCopy.begin());

    auto start = std::chrono::steady_clock::now();

    float* d_tIn, * d_pIn, * d_tOut;
    cudaMalloc((void**)&d_tIn, size * sizeof(float));
    cudaMalloc((void**)&d_pIn, size * sizeof(float));
    cudaMalloc((void**)&d_tOut, size * sizeof(float));

    cudaMemcpy(d_tIn, tIn.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pIn, pIn.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    for (int j = 0; j < iterations; j++) {
        computeTemperatureKernel<<<gridSize, blockSize>>>(d_tIn, d_pIn, d_tOut, size, Cap, Rx, Ry, Rz, dt, amb_temp);
        std::swap(d_tIn, d_tOut);
    }

    cudaMemcpy(tIn.data(), d_tIn, size * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (duration * 1e-3f) / iterations << " (us)" << std::endl;

    if (iterations & 01) {
        sel = tIn.data();
    } else {
        sel = tOut.data();
    }

    auto stop = std::chrono::steady_clock::now();
    auto deviceTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << "Device offloading time: " << deviceTime / 1000000.0 << " (s)" << std::endl;

    float* answer = new float[size];
    computeTempCPU(pIn.data(), tCopy.data(), answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

    float acc = accuracy(sel, answer, size);
    std::cout << "Root-mean-square error: " << acc << std::endl;

    writeoutput(tIn.data(), numRows, numCols, layers, ofile);

    delete[] answer;
    cudaFree(d_tIn);
    cudaFree(d_pIn);
    cudaFree(d_tOut);
    return 0;
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, const char *file) {
    int i, j, k;
    std::ifstream fp(file);
    if (!fp.is_open()) {
        std::cerr << "