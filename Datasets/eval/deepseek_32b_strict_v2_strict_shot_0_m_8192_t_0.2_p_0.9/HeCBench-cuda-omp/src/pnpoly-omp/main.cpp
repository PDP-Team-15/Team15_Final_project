#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>

#define VERTICES 600

typedef struct {
    float x;
    float y;
} float2;

void pnpoly_base(int* bitmap, const float2* points, const float2* vertices, int nPoints, int nVertices) {
    #pragma omp parallel for
    for (int i = 0; i < nPoints; ++i) {
        const float2 p = points[i];
        int crossings = 0;
        for (int j = 0; j < nVertices; ++j) {
            const float2 v1 = vertices[j];
            const float2 v2 = vertices[(j + 1) % nVertices];
            if (p.x > fmin(v1.x, v2.x) && p.x <= fmax(v1.x, v2.x)) {
                float t = (p.x - v1.x) / (v2.x - v1.x);
                float y = v1.y + t * (v2.y - v1.y);
                if (p.y == y) {
                    crossings++;
                } else if (p.y < y) {
                    if (v1.y > v2.y) {
                        crossings++;
                    }
                } else {
                    if (v1.y < v2.y) {
                        crossings++;
                    }
                }
            }
        }
        bitmap[i] = crossings % 2;
    }
}

void pnpoly_opt(int* bitmap, const float2* points, const float2* vertices, int nPoints, int nVertices, int opt) {
    #pragma omp parallel for
    for (int i = 0; i < nPoints; ++i) {
        const float2 p = points[i];
        int crossings = 0;
        for (int j = 0; j < nVertices; j += opt) {
            for (int k = 0; k < opt; ++k) {
                int idx = j + k;
                if (idx >= nVertices) break;
                const float2 v1 = vertices[idx];
                const float2 v2 = vertices[(idx + 1) % nVertices];
                if (p.x > fmin(v1.x, v2.x) && p.x <= fmax(v1.x, v2.x)) {
                    float t = (p.x - v1.x) / (v2.x - v1.x);
                    float y = v1.y + t * (v2.y - v1.y);
                    if (p.y == y) {
                        crossings++;
                    } else if (p.y < y) {
                        if (v1.y > v2.y) {
                            crossings++;
                        }
                    } else {
                        if (v1.y < v2.y) {
                            crossings++;
                        }
                    }
                }
            }
        }
        bitmap[i] = crossings % 2;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: ./%s <repeat>\n", argv[0]);
        return 1;
    }

    const int repeat = atoi(argv[1]);
    const int nPoints = 20000000;
    const int vertices = VERTICES;

    std::default_random_engine rng(123);
    std::normal_distribution<float> distribution(0, 1);

    float2* point = (float2*)malloc(nPoints * sizeof(float2));
    for (int i = 0; i < nPoints; ++i) {
        point[i].x = distribution(rng);
        point[i].y = distribution(rng);
    }

    float2* vertex = (float2*)malloc(vertices * sizeof(float2));
    for (int i = 0; i < vertices; ++i) {
        float t = distribution(rng) * 2.f * M_PI;
        vertex[i].x = cosf(t);
        vertex[i].y = sinf(t);
    }

    int* bitmap_ref = (int*)malloc(nPoints * sizeof(int));
    int* bitmap_opt = (int*)malloc(nPoints * sizeof(int));

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_base(bitmap_ref, point, vertex, nPoints, vertices);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_opt(bitmap_opt, point, vertex, nPoints, vertices, 1);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_opt(bitmap_opt, point, vertex, nPoints, vertices, 2);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_opt(bitmap_opt, point, vertex, nPoints, vertices, 4);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_opt(bitmap_opt, point, vertex, nPoints, vertices, 8);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_opt(bitmap_opt, point, vertex, nPoints, vertices, 16);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_opt(bitmap_opt, point, vertex, nPoints, vertices, 32);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
        pnpoly_opt(bitmap_opt, point, vertex, nPoints, vertices, 64);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

    int error = memcmp(bitmap_opt, bitmap_ref, nPoints * sizeof(int));

    int checksum = 0;
    for (int i = 0; i < nPoints; ++i) checksum += bitmap_opt[i];
    printf("Checksum: %d\n", checksum);

    printf("%s\n", error ? "FAIL" : "PASS");

    free(point);
    free(vertex);
    free(bitmap_ref);
    free(bitmap_opt);

    return 0;
}