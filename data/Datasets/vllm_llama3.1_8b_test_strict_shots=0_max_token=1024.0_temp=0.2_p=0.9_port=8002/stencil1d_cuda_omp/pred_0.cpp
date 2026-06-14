```cpp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define RADIUS 7
#define BLOCK_SIZE 256

void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int length) {
  int gindex, lindex, offset, result;
  int temp[BLOCK_SIZE + 2 * RADIUS];

  #pragma omp parallel for default(none) shared(in, out, temp) num_threads(BLOCK_SIZE)
  for (gindex = 0; gindex < length; gindex++) {
    lindex = gindex + RADIUS;
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
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int size = length * sizeof(int);
  int pad_size = (length + RADIUS) * sizeof(int);

  int *a, *b;

  a = (int *)malloc(pad_size);
  b = (int *)malloc(size);

  for (int i = 0; i < length+RADIUS; i++) a[i] = i;

  for (int i = 0; i < repeat; i++) {
    stencil_1d(a, b, length);
  }

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