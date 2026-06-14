#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define RADIUS 7

void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int length)
{
  #pragma omp parallel
  {
    int block_size = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    int gindex = thread_id + block_size * (length / block_size);
    int lindex = thread_id + RADIUS;

    int temp[block_size + 2 * RADIUS];

    temp[lindex] = in[gindex];

    if (thread_id < RADIUS) {
      temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
      temp[lindex + block_size] = in[gindex + block_size];
    }

    #pragma omp barrier

    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[lindex + offset];

    out[gindex] = result;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", omp_get_max_threads());
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int size = length * sizeof(int);
  int pad_size = (length + RADIUS) * sizeof(int);

  int *a, *b;

  a = (int *)malloc(pad_size);
  b = (int *)malloc(size);

  for (int i = 0; i < length + RADIUS; i++) a[i] = i;

  int *d_a, *d_b;

  d_a = (int *)malloc(pad_size);
  d_b = (int *)malloc(size);

  for (int i = 0; i < pad_size / sizeof(int); i++) d_a[i] = a[i];

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    stencil_1d(d_a, d_b, length);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  for (int i = 0; i < size / sizeof(int); i++) b[i] = d_b[i];

  bool ok = true;
  for (int i = 0; i < 2 * RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i + 2 * RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2 * RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i - RADIUS; j <= i + RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(a);
  free(b);
  free(d_a);
  free(d_b);
  return 0;
}