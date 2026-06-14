You are an HPC expert specializing in translating between parallel programming APIs. You will translate OMP code to CUDA code with the following requirements:

1. Include all necessary CUDA headers (e.g., cuda_runtime.h, etc.)
2. Replace OMP function calls with correct CUDA equivalents
3. Use proper CUDA syntax for kernel launches and memory management
4. Ensure all variables are correctly declared and accessible within CUDA context
5. Function names must match the original code
6. Include complete code only - no partial lines or commented code
7. Output must compile without errors on first attempt
8. CUDA kernel functions must be marked with __global__ 
9. Helper functions must be marked with __device__
10. Include all necessary includes and preprocessor directives
11. Ensure proper memory alignment and indexing
12. Include complete error checking for CUDA API calls (if applicable)
13. Use standard CUDA best practices for code structure and style
14. All code must be syntactically complete - no missing semicolons or parenthesis
15. Preserve the original algorithm's logic and functionality

Output only the translated code inside a code block. No explanations, no comments, no omissions. Ensure all CUDA-specific syntax is properly formatted and complete.
For each kernel code provided, translate it from cuda to omp. Provide the complete code in omp. Do not truncate or use ellipses. Ensure correctness. Do not add any additional comments or explanations.
Translate the following code  from cuda to omp:
main.cu:
#include <chrono>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>
#include <cuda.h>

#define HOSTDEVICE __host__ __device__



#define BSIZE 256

template <class T>
class AvgPoolGrad {
  public:
    HOSTDEVICE inline void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
      *dx += (scale * dy);
    }
};

template <class T>
class MaxPoolGrad {
  public:
    HOSTDEVICE inline void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
      *dx += dy * static_cast<T>(x == y);
    }
};

#include "reference.h"

template <typename PoolProcess, typename T,
          int ksize_height,
          int ksize_width,
          int stride_height,
          int stride_width,
          int padding_height,
          int padding_width,
          bool exclusive>
__global__ void KernelPool2DGrad(
    const int nthreads,
    const T*__restrict__ input_data,
    const T*__restrict__ output_data,
    const T*__restrict__ output_grad,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    PoolProcess pool_process,
    T*__restrict__ input_grad,
    bool channel_last = false)
{
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
           index += blockDim.x * gridDim.x) {
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
    phend = min(h_offset / stride_height + 1, output_height);
    pwend = min(w_offset / stride_width + 1, output_width);

    

    T gradient = static_cast<T>(0.0);
    T input = input_data[index];

    int output_stride = batch_idx * output_height * output_width * channels;
    if (!channel_last)
      output_stride += offsetC * output_height * output_width;

    const T *__restrict__ output_data_t = output_data + output_stride;
    const T *__restrict__ output_grad_t = output_grad + output_stride;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int pool_size;
        int hstart = ph * stride_height - padding_height;
        int wstart = pw * stride_width - padding_width;
        int hend = min(hstart + ksize_height, input_height);
        int wend = min(wstart + ksize_width, input_width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
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
}

int main(int argc, char* argv[])
{
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

  const int input_numel = batch_size*input_channels*input_height*input_width;
  const int output_numel = batch_size*input_channels*output_height*output_width;

  

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

  float * input = new float[input_numel];
  float * output = new float[output_numel];
  float * output_grad = new float[output_numel];
  float * input_grad = new float[input_numel];
  float * input_grad_ref = new float[input_numel];

  srand(123);
  for (int i = 0; i < input_numel; ++i) {
    input[i] = (float)rand() / (float)RAND_MAX;
    input_grad[i] = 0.f;  

  }

  for (int i = 0; i < output_numel; ++i) {
    output[i] = (float)rand() / (float)RAND_MAX;
    output_grad[i] = input_width * input_height;
  }

  float *input_data, *output_data, *output_grad_data, *input_grad_data;
  cudaMalloc((void **)&input_data, input_numel * sizeof(float));
  cudaMalloc((void **)&input_grad_data, input_numel * sizeof(float));
  cudaMalloc((void **)&output_data, output_numel * sizeof(float));
  cudaMalloc((void **)&output_grad_data, output_numel * sizeof(float));
  cudaMemcpy(input_data, input, input_numel * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_data, output, output_numel * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_grad_data, output_grad, output_numel * sizeof(float), cudaMemcpyHostToDevice);

  int blocks = (nthreads + BSIZE - 1) / BSIZE;
  dim3 threads(BSIZE);
  dim3 grid(blocks);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    KernelPool2DGrad<AvgPoolGrad<float>, float, ksize_height, ksize_width,
                     stride_height, stride_width, padding_height, padding_width,
                     exclusive><<<grid, threads, 0, 0>>>(
        nthreads, input_data, output_data, output_grad_data, input_channels,
        input_height, input_width, output_height, output_width, pool_process,
        input_grad_data, channel_last);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(input_grad, input_grad_data, input_numel * sizeof(float), cudaMemcpyDeviceToHost);

  

  reference<AvgPoolGrad<float>, float>(
          nthreads, input, output, output_grad,
          input_channels, input_height, input_width, output_height, output_width, ksize_height,
          ksize_width, stride_height, stride_width, padding_height, padding_width,
          pool_process, exclusive, input_grad_ref, channel_last);

  bool ok = true;
  for (int i = 0; i < input_numel; ++i) {
    if (fabsf(input_grad[i] - input_grad_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] input;
  delete[] output;
  delete[] input_grad;
  delete[] input_grad_ref;
  delete[] output_grad;
  cudaFree(input_data);
  cudaFree(input_grad_data);
  cudaFree(output_data);
  cudaFree(output_grad_data);
  return 0;
}
