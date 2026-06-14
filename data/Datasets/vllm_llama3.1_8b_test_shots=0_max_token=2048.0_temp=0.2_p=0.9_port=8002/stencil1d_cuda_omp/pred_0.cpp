Here's the equivalent code in OpenMP:

```c
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define RADIUS 7
#define BLOCK_SIZE 256

void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int length) {
  int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex, lindex, offset, i;

  #pragma omp parallel for default(none) shared(temp, in, out, length) private(gindex, lindex, offset, i)
  for (gindex = 0; gindex < length; gindex++) {
    lindex = gindex + RADIUS;

    temp[lindex] = in[gindex];

    if (gindex < RADIUS) {
      temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    #pragma omp barrier

    int result = 0;
    for (offset = -RADIUS; offset <= RADIUS; offset++) {
      result += temp[lindex + offset];
    }

    out[gindex] = result;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  if (length % BLOCK_SIZE != 0) {
    printf("Error: length must be a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }

  int *a, *b;
  a = (int *)malloc((length + RADIUS) * sizeof(int));
  b = (int *)malloc(length * sizeof(int));

  for (int i = 0; i < length + RADIUS; i++) a[i] = i;

  int time = 0;

  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    stencil_1d(a, b, length);
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  time /= repeat;

  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f));

  bool ok = true;
  for (int i = 0; i < 2*RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i+2*RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2*RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i-RADIUS; j <= i+RADIUS; j++)
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
  return 0;
}
```

Note that OpenMP does not have a direct equivalent to CUDA's `__global__` kernel launch, so we use the `#pragma omp parallel for` directive to parallelize the loop. Also, we use the `omp_set_num_threads` function to set the number of threads, but in this case, we don't need to do that because the number of threads is determined by the number of available cores.

Also, note that OpenMP does not have a direct equivalent to CUDA's `__shared__` memory, so we use a regular array to store the temporary values.