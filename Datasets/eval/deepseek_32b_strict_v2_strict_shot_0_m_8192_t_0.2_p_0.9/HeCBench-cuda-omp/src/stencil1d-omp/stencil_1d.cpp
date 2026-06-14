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

void stencil_1d(const int *in, int *out, int length) {
    int num_threads = BLOCK_SIZE;
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        int temp[BLOCK_SIZE + 2 * RADIUS];
        
        int start = tid * (length / total_threads);
        int end = (tid + 1) * (length / total_threads);
        
        for (int i = start; i < end; i++) {
            int lindex = tid + RADIUS;
            temp[lindex] = in[i];
            
            if (tid < RADIUS) {
                if (i - RADIUS >= 0) {
                    temp[lindex - RADIUS] = in[i - RADIUS];
                } else {
                    temp[lindex - RADIUS] = 0;
                }
                if (i + BLOCK_SIZE < length) {
                    temp[lindex + BLOCK_SIZE] = in[i + BLOCK_SIZE];
                } else {
                    temp[lindex + BLOCK_SIZE] = 0;
                }
            }
        }
        
        #pragma omp barrier
        
        for (int i = start; i < end; i++) {
            int lindex = tid + RADIUS;
            int result = 0;
            for (int offset = -RADIUS; offset <= RADIUS; offset++) {
                result += temp[lindex + offset];
            }
            out[i] = result;
        }
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
    
    for (int i = 0; i < length + RADIUS; i++) {
        a[i] = i;
    }
    
    int *d_a = a;
    int *d_b = b;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < repeat; i++) {
        stencil_1d(d_a, d_b, length);
    }
    
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
    
    bool ok = true;
    for (int i = 0; i < 2 * RADIUS; i++) {
        int s = 0;
        for (int j = i; j <= i + 2 * RADIUS; j++) {
            s += (j < RADIUS) ? 0 : (a[j] - RADIUS);
        }
        if (s != b[i]) {
            printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
            ok = false;
            break;
        }
    }
    
    for (int i = 2 * RADIUS; i < length; i++) {
        int s = 0;
        for (int j = i - RADIUS; j <= i + RADIUS; j++) {
            s += a[j];
        }
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