```cpp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define RADIUS 7
#define BLOCK_SIZE 256

void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int length) {
  int *temp = (int *)malloc((BLOCK_SIZE + 2 * RADIUS) * sizeof(int));
  int gindex, lindex, offset, result;

  #pragma omp parallel for
  for (int i = 0; i < length; i++) {
    gindex = i;
    lindex = i + RADIUS;

    temp[lindex] = in[gindex];

    if (omp_get_thread_num() < RADIUS) {
      temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    #pragma omp barrier

    result = 0;
    for (offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[lindex + offset];

    out[gindex] = result;
  }

  free(temp);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int *a, *b;

  a = (int *)malloc((length + RADIUS) * sizeof(int));
  b = (int *)malloc(length * sizeof(int));

  for (int i = 0; i < length+RADIUS; i++) a[i] = i;

  int *d_a, *d_b;

  d_a = (int *)malloc((length + RADIUS) * sizeof(int));
  d_b = (int *)malloc(length * sizeof(int));

  for (int i = 0; i < length+RADIUS; i++) d_a[i] = a[i];

  int time = 0;

  for (int i = 0; i < repeat; i++) {
    #pragma omp parallel
    {
      #pragma omp master
      {
        auto start = omp_get_wtime();
      }
      #pragma omp barrier
      stencil_1d(a, d_b, length);
      #pragma omp master
      {
        auto end = omp_get_wtime();
        time += (end - start);
      }
    }
  }

  printf("Average kernel execution time: %f (s)\n", (time * 1.0 / repeat) / omp_get_num_threads());

  for (int i = 0; i < length; i++) b[i] = d_b[i];

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
  free(d_a);
  free(d_b);
  return 0;
}
```