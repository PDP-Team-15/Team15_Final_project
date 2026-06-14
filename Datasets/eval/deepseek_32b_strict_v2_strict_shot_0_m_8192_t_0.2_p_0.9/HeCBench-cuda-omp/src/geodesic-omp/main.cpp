#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <omp.h>

struct float4 {
    float x, y, z, w;
};

float distance_host(int i, float latitude_1, float longitude_1,
                   float latitude_2, float longitude_2) {
    // ... (same as original code)
}

void kernel_distance(const float4* d_A, float* d_C, const int N) {
    #pragma omp parallel
    {
        int wiID = omp_get_thread_num();
        if (wiID >= N) return;

        // ... (same as original kernel code)
    }
}

void distance_device(const float4* VA, float* VC, const size_t N, const int iteration) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            float4* d_VA = (float4*)malloc(N * sizeof(float4));
            float* d_VC = (float*)malloc(N * sizeof(float));

            memcpy(d_VA, VA, N * sizeof(float4));
            
            double start = omp_get_wtime();
            for (int n = 0; n < iteration; n++) {
                kernel_distance(d_VA, d_VC, N);
            }
            double end = omp_get_wtime();
            
            printf("Average kernel execution time %f (s)\n", (end - start) / iteration);
            
            memcpy(VC, d_VC, N * sizeof(float));
            
            free(d_VA);
            free(d_VC);
        }
    }
}

void verify(int size, const float* output, const float* expected_output) {
    float error_rate = 0;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        if (fabs(output[i] - expected_output[i]) > error_rate) {
            error_rate = fabs(output[i] - expected_output[i]);
        }
    }
    printf("The maximum error in distance for single precision is %f\n", error_rate);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage %s <repeat>\n", argv[0]);
        return 1;
    }
    int iteration = atoi(argv[1]);

    int num_cities = 2097152;
    int num_ref_cities = 6;
    int index_map[] = {436483, 1952407, 627919, 377884, 442703, 1863423};
    int N = num_cities * num_ref_cities;
    int city = 0;
    float lat, lon;

    const char* filename = "locations.txt";
    printf("Reading city locations from file %s...\n", filename);
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening the file");
        exit(-1);
    }

    float4* input = (float4*)aligned_alloc(4096, N * sizeof(float4));
    float* output = (float*)aligned_alloc(4096, N * sizeof(float));
    float* expected_output = (float*)malloc(N * sizeof(float));

    while (fscanf(fp, "%f %f\n", &lat, &lon) != EOF) {
        input[city].x = lat;
        input[city].y = lon;
        city++;
        if (city == num_cities) break;
    }
    fclose(fp);

    for (int c = 1; c < num_ref_cities; c++) {
        std::copy(input, input + num_cities, input + c * num_cities);
    }

    for (int c = 0; c < num_ref_cities; c++) {
        int index = index_map[c] - 1;
        for (int j = c * num_cities; j < (c + 1) * num_cities; ++j) {
            input[j].z = input[index].x;
            input[j].w = input[index].y;
        }
    }

    for (int i = 0; i < N; i++) {
        float lat1 = input[i].x;
        float lon1 = input[i].y;
        float lat2 = input[i].z;
        float lon2 = input[i].w;
        expected_output[i] = distance_host(i, lat1, lon1, lat2, lon2);
    }

    distance_device(input, output, N, iteration);

    verify(N, output, expected_output);

    free(input);
    free(output);
    free(expected_output);
    return 0;
}