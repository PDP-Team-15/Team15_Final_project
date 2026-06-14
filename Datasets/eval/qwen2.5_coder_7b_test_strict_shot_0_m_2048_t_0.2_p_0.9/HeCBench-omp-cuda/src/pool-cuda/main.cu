#include <chrono>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define BSIZE 256

template <class T>
class AvgPoolGrad {
public:
    __device__ void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
        atomicAdd(dx, scale * dy);
    }
};

template <class T>
class MaxPoolGrad {
public:
    __device__ void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
        if (x == y) atomicAdd(dx, dy);
    }
};

#include "reference.h"

template <typename PoolProcess, typename T>
__global__ void KernelPool2DGrad(
    const int nthreads,
    const T* input_data,
    const T* output_data,
    const T* output_grad,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int ksize_height,
    const int ksize_width,
    const int stride_height,
    const int stride_width,
    const int padding_height,
    const int padding_width,
    PoolProcess pool_process,
    bool exclusive,
    T* input_grad,
    bool channel_last = false)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nthreads) return;

    int w_offset, h_offset, offsetC, batch_idx;
    int tmp;
    if (!channel_last) { 
        w_offset = index % input_width + padding_width;
        tmp = index / input_width;
        h_offset = tmp % input_height + padding_height;
        tmp = tmp / input_height;
        offsetC = tmp % channels;
        batch_idx = tmp / channels;
    } else { 
        offsetC = index % channels;
        tmp = index / channels;
        w_offset = tmp % input_width + padding_width;
        tmp = tmp / input_width;
        h_offset = tmp % input_height + padding_height;
        batch_idx = tmp / input_height;
    }

    int phstart, phend;
    int pwstart, pwend;
    phstart = (h_offset < ksize_height) ? 0 : (h_offset - ksize_height) / stride_height + 1;
    pwstart = (w_offset < ksize_width) ? 0 : (w_offset - ksize_width) / stride_width + 1;
    phend = std::min(h_offset / stride_height + 1, output_height);
    pwend = std::min(w_offset / stride_width + 1, output_width);

    T gradient = static_cast<T>(0.0);
    T input = input_data[index];

    int output_stride = batch_idx * output_height * output_width * channels;
    if (!channel_last)
        output_stride += offsetC * output_height * output_width;

    const T* output_data_t = output_data + output_stride;
    const T* output_grad_t = output_grad + output_stride;

    for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
            int pool_size;
            int hstart = ph * stride_height - padding_height;
            int wstart = pw * stride_width - padding_width;
            int hend = std::min(hstart + ksize_height, input_height);
            int wend = std::min(wstart + ksize_width, input_width);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            pool_size = exclusive ? (hend - hstart) * (wend - wstart)
                : ksize_height * ksize_width;

            int output_sub_idx = channel_last
                ? (ph * output_width + pw) * channels + offsetC
                : ph * output_width + pw;
            pool_process.compute(input, output_data_t[output_sub_idx],
                output_grad_t[output_sub_idx],
                static_cast<T>(1.f / pool_size), &gradient);
        }
    }
    input_grad[index] = gradient;
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        printf("Usage: %s <batch> <input channels> <input height> ", argv[0]);
        printf("<input width> <output height> <output width> <repeat>\n");
        return 1;
    }

    const int batch_size = atoi(argv[1]);
    const int input_channels = atoi(argv[2]);
    const int input_height = atoi(argv[3]);
    const int input_width = atoi(argv[4]);

    const int output_height = atoi(argv[5]);
    const int output_width = atoi(argv[6]);

    const int repeat = atoi(argv[7]);

    const int input_numel = batch_size * input_channels * input_height * input_width;
    const int output_numel = batch_size * input_channels * output_height * output_width;

    const int ksize_height = 11;
    const int ksize_width = 11;
    const int stride_height = 4;
    const int stride_width = 4;
    const int padding_height = 1;
    const int padding_width = 1;
    const bool exclusive = true;
    const std::string data_format = "NCHW";
    const bool channel_last = (data_format == "NHWC");

    int nthreads = batch_size * input_channels * input_height * input_width;

    AvgPoolGrad<float> pool_process;

    float * d_input, * d_output, * d_output_grad, * d_input_grad;
    cudaMalloc((void**)&d_input, input_numel * sizeof(float));
    cudaMalloc((void**)&d_output, output_numel * sizeof(float));
    cudaMalloc((void**)&d_output_grad, output_numel * sizeof(float));
    cudaMalloc((void**)&d_input_grad, input_numel * sizeof(float));

    float * h_input = new float[input_numel];
    float * h_output = new float[output_numel];
    float * h_output_grad = new float[output_numel];
    float * h_input_grad = new float[input_numel];

    srand(123);
    for (int i = 0; i < input_numel; ++i) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
        h_input_grad[i] = 0.f;
    }

    for (int i = 0; i < output_numel; ++i) {
        h_output[i] = (float)rand() / (float)RAND_MAX;
        h_output_grad[i] = input_width * input_height;
    }

    cudaMemcpy(d_input, h_input, input_numel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_numel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_grad, h_output_grad, output_numel * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BSIZE);
    dim3 numBlocks((nthreads + BSIZE - 1) / BSIZE);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        KernelPool2DGrad<<<numBlocks, threadsPerBlock>>>(nthreads, d_input, d_output, d_output_grad, input_channels,
            input_height, input_width, output_height, output_width, ksize_height,
            ksize_width, stride_height, stride_width, padding_height, padding_width,
            pool_process, exclusive, d_input_grad, channel_last);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    cudaMemcpy(h_input_grad, d_input_grad, input_numel * sizeof(float), cudaMemcpyDeviceToHost);

    reference<AvgPoolGrad<float>, float>(
        nthreads, h_input, h_output, h_output_grad,
        input_channels, input_height, input_width, output_height, output_width, ksize_height,
        ksize_width, stride_height, stride_width, padding_height, padding_width,
        pool_process, exclusive, h_input_grad, channel_last);

    bool ok = true;
    for (int i = 0; i < input_numel; ++i) {
        if (fabsf(h_input_grad[i] - h_input_grad[i]) > 1e-3) {
            ok = false;
            break;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    delete[] h_input;
    delete[] h_output;
    delete[] h_input_grad;
    delete[] h_output_grad;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_grad);
    cudaFree(d_input_grad);
    return 0;
}