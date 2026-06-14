#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define RADIUS 7

void stencil_1d(const int *in, int *out, int length) {
  #pragma omp parallel for
  for (int gindex = 0; gindex < length; ++gindex) {
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
      int lindex = gindex + offset;
      if (lindex >= 0 && lindex < length) {
        result += in[lindex];
      } else if (lindex < 0) {
        result += 0;
      } else {
        // Handle boundary conditions as needed
        result += in[length - 1]; // Example: wrap around
      }
    }
    out[gindex] = result;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", 1);
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
  
  d_a = a;
  d_b = b;

  double total_time = 0.0;

  for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    stencil_1d(d_a, d_b, length);
    auto end = std::chrono::steady_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time: %f (s)\n", (total_time * 1e-9f) / repeat);

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
  return 0;
}