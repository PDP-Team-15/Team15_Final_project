#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define RADIUS 7
#define BLOCK_SIZE 256

void stencil_1d(int* in, int* out, int length, int RADIUS) {
  int num_threads = omp_get_max_threads();
  int* local_in = (int*)malloc((BLOCK_SIZE + 2 * RADIUS) * sizeof(int));
  
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int block_start = thread_id * BLOCK_SIZE;
    int block_end = block_start + BLOCK_SIZE - 1;
    
    if (block_end >= length) {
      block_end = length - 1;
    }
    
    for (int i = 0; i < BLOCK_SIZE + 2 * RADIUS; i++) {
      int global_index = block_start - RADIUS + i;
      local_in[i] = (global_index < 0 || global_index >= length) ? 0 : in[global_index];
    }
    
    #pragma omp barrier
    
    for (int i = block_start; i <= block_end; i++) {
      int result = 0;
      for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        int lindex = i - block_start + RADIUS + offset;
        result += local_in[lindex];
      }
      out[i] = result;
    }
  }
  
  free(local_in);
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

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    stencil_1d(a, b, length, RADIUS);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

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