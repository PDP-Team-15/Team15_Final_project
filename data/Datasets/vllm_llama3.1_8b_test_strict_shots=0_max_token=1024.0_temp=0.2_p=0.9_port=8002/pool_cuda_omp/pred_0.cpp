```cpp
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>
#include <iostream>

#define HOSTDEVICE inline
#define BSIZE 256

template <class T>
class AvgPoolGrad {
  public:
    HOSTDEVICE void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
      *dx += (scale * dy);
    }
};

template <class T>
class MaxPoolGrad {
  public:
    HOSTDEVICE void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
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
void KernelPool2DGrad(
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
  #pragma omp parallel for
  for (int index = 0; index < nthreads; ++index) {
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
    std::printf("Usage: %s <batch> <input channels> <input height> ", argv[0]);
    std::printf("<input width> <output height> <output width> <repeat>\n");
    return 1;
  }

  const int batch_size = std::atoi(argv[1]);
  const int input_channels = std::atoi(argv[2]);
  const int input_height = std::atoi(argv[3]);
  const int input_width = std::atoi(argv