```cpp
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>

#define BSIZE 256

template <class T>
__global__ void KernelPool2DGradKernel(
    const T*__restrict input_data,
    const T*__restrict output_data,
    const T*__restrict output_grad,
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
    T*__restrict input_grad,
    bool channel_last)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= input_height * input_width * channels) return;

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

  const T *__restrict output_data_t = output_data + output_stride;
  const T *__restrict output_grad_t = output_grad + output_stride;

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

template <typename PoolProcess, typename T>
void KernelPool2DGrad(
    const int nthreads,
    const T*__restrict input_data,
    const T*__restrict output_data,
    const T*__restrict output_grad,
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
    T*__restrict input_grad,
    bool channel_last = false)
{
  int blockSize = BSIZE;
  int numBlocks = (nthreads + blockSize - 1) / blockSize;
  KernelPool2DGradKernel<<<numBlocks, blockSize>>>(input_data, output_data, output_grad, channels, input_height, input_width, output_height, output_width, ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width, pool_process, exclusive, input_grad, channel_last);
  cudaError_t err = cuda