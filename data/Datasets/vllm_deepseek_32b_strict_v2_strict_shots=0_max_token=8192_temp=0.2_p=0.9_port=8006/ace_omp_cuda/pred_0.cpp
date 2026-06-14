<think>
Okay, so I need to translate this OpenMP code into CUDA. Let me start by understanding what the original code does. It's a simulation involving some PDEs, probably related to phase transitions given the Allen-Cahn equation mention. The code has several functions: calculateForce, allenCahn, boundaryConditionsPhi, thermalEquation, boundaryConditionsU, swapGrid, initializationPhi, initializationU, and the main function.

First, I'll look at the structure. The main function initializes arrays, runs the simulation for a number of steps, and then verifies the results. The functions are using OpenMP for parallelization, especially with target teams distribute parallel for collapse(3) directives, which suggests they're offloading work to a GPU.

To translate this to CUDA, I need to:

1. Replace OpenMP directives with CUDA kernel launches.
2. Manage memory on the GPU using cudaMemcpy and device pointers.
3. Ensure that all functions are correctly translated, especially those marked with #pragma omp declare target, which in CUDA would be __device__ functions.
4. Handle the data transfer between host and device, and ensure that all necessary data is on the GPU when needed.

Looking at the functions:

- calculateForce: This function computes forces based on gradients and other parameters. It uses collapse(3) for parallelization, so in CUDA, this will be a kernel that each thread handles a specific (ix, iy, iz) point.

- allenCahn: Updates the phase field using the Allen-Cahn equation. Similarly, this will be a kernel.

- boundaryConditionsPhi and boundaryConditionsU: Apply boundary conditions. These can be kernels as well, or perhaps handled within the main computation.

- thermalEquation: Computes the temperature field. Another kernel.

- swapGrid: Swaps two grids. In CUDA, this might involve simply swapping pointers or using cudaMemcpy, but since it's a parallel operation, it's better to handle it with a kernel or by managing pointers.

- initializationPhi and initializationU: Initialize the grids. These can be done with CUDA kernels or using memset on the GPU.

Now, considering the data structures. The original code uses nRarray, which is a 2D array, but in 3D space, so each phi is a 3D array. In CUDA, I'll need to manage these as 3D arrays, possibly using pitched pointers or 1D arrays with calculated indices.

Memory management is crucial. I'll need to allocate memory on the GPU for all the necessary arrays: phiold, phinew, uold, unew, Fx, Fy, Fz. Each of these will be a pointer to device memory.

In the main function, the original code uses target data map to transfer data to the GPU. In CUDA, this translates to cudaMemcpy. The initial data (phi_host and u_host) are on the host, so I'll need to copy them to the device.

The main loop runs for num_steps, each step involving several function calls. Each of these functions will be translated into CUDA kernels. The functions that were using target teams distribute parallel for collapse(3) will become kernels with thread blocks arranged in 3D.

I also need to handle the swapGrid function. In CUDA, instead of swapping data, which is expensive, I can swap pointers. So, I'll have two sets of pointers for phi and u, and just switch which ones are considered new and old each iteration.

Another consideration is error checking. CUDA requires checking the return values of CUDA API calls to ensure there are no errors. I'll need to add error checking for each cudaMemcpy, kernel launch, and memory allocation.

Looking at the functions:

- calculateForce: Computes Fx, Fy, Fz based on gradients. The gradients are computed using finite differences. In CUDA, each thread will compute the gradients for its specific (ix, iy, iz) point.

- allenCahn: Uses the computed forces and the old phi and u to compute the new phi. Each thread handles one point.

- boundaryConditionsPhi and boundaryConditionsU: These set the boundaries to specific values. In CUDA, these can be handled as kernels, but since the boundaries are the same for all points, perhaps it's more efficient to handle them within the main computation or use a separate kernel.

- thermalEquation: Computes the new u based on the old u and phi. Again, a kernel.

- swapGrid: As mentioned, this can be handled by swapping pointers.

Now, considering the data transfer. The initial phi and u are on the host, so I'll need to copy them to the device. After the simulation, the results are copied back to the host for verification.

In the original code, the functions are called within the target data region, which in CUDA translates to managing the data on the device. So, all the function calls will be within the CUDA context, meaning the data must be on the device.

I also need to consider the function signatures. In CUDA, device functions (marked with __device__) cannot return pointers or have certain types. So, I'll need to adjust the function signatures to fit CUDA's requirements.

Another point is the use of shared memory or registers for performance. However, since the functions are relatively simple, using global memory might be sufficient, but I can consider optimizing later.

Now, let's think about the main steps:

1. Include necessary CUDA headers: cuda_runtime.h, etc.

2. Replace OpenMP function calls with CUDA equivalents. For example, the target teams distribute parallel for collapse(3) becomes a kernel launch with a 3D grid.

3. Use proper CUDA syntax for kernel launches, including the <<<grid, block>>> syntax.

4. Ensure all variables are correctly declared and accessible. For example, device pointers must be used for data on the GPU.

5. Function names must match the original code, but with __global__ or __device__ qualifiers.

6. Include complete code without partial lines or comments.

7. Ensure the code compiles without errors. This means proper error checking and correct syntax.

8. CUDA kernel functions must be marked with __global__, and helper functions with __device__.

9. Include all necessary includes and preprocessor directives.

10. Ensure proper memory alignment and indexing. For 3D arrays, using a single index calculated from x, y, z is common.

11. Include complete error checking for CUDA API calls.

12. Use standard CUDA best practices, such as using streams for overlapping computation and memory transfers, but for simplicity, perhaps not in this initial translation.

13. Ensure all code is syntactically complete.

14. Preserve the original algorithm's logic and functionality.

Now, let's start translating each function.

First, the helper functions like dFphi, GradientX, etc., which were declared with #pragma omp declare target, will become __device__ functions in CUDA.

The main functions like calculateForce, allenCahn, etc., will become __global__ functions.

In the main function, the data is allocated on the host and then copied to the device. The main loop will launch each kernel, followed by necessary memory transfers.

For example, the calculateForce function in OpenMP is a target function, so in CUDA, it becomes a kernel. Each thread will handle a specific (ix, iy, iz) point.

I'll need to adjust the loop indices to match the kernel's thread indices. In CUDA, each thread is identified by blockIdx and threadIdx, so the global index is calculated as blockIdx.x * blockDim.x + threadIdx.x, but for 3D, it's more complex.

Alternatively, I can use a 3D grid where each thread corresponds to a specific (x, y, z) point. So, the kernel will have:

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz, ...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        // compute Fx, Fy, Fz
    }
}

But I need to ensure that the grid dimensions cover the entire data space. So, the kernel launch will have a grid size of (DATAXSIZE, DATAYSIZE, DATAZSIZE) and a block size that's efficient, perhaps 8x8x8 or similar.

Wait, but in the original code, the loops are from 0 to DATAXSIZE-1, etc. So, the kernel needs to cover all these points.

Another consideration is that in CUDA, the maximum block size is limited, so for large DATAXSIZE, I might need to adjust the block and grid sizes accordingly.

Now, considering the main loop:

Each iteration involves:

1. calculateForce: Launches a kernel to compute Fx, Fy, Fz.

2. allenCahn: Launches a kernel to compute phinew.

3. boundaryConditionsPhi: Applies boundary conditions to phinew.

4. thermalEquation: Computes unew.

5. boundaryConditionsU: Applies boundary conditions to unew.

6. swapGrid: Swaps phiold and phinew, uold and unew.

In CUDA, each of these steps will involve launching a kernel, except for swapGrid, which can be handled by swapping pointers.

But wait, in the original code, swapGrid is a target function, so in CUDA, it would be a kernel. However, swapping pointers is more efficient, so perhaps I can manage it without a kernel.

Alternatively, if the swapGrid function is simple, it can be a kernel that copies data, but that's inefficient. So, better to manage with pointers.

So, I'll have two sets of pointers for phi and u: phi_old, phi_new, u_old, u_new. Each iteration, after computing phi_new and u_new, I'll swap the pointers so that phi_old becomes phi_new and vice versa.

This way, I avoid copying data and just swap the pointers, which is efficient.

Now, considering the data transfer:

The initial phi and u are on the host, so I'll need to copy them to the device. After the simulation, I'll copy them back.

In the main function, after allocating device memory, I'll perform cudaMemcpy from the host to the device.

Now, let's outline the steps in the main function:

1. Read command-line arguments for num_steps.

2. Initialize constants: dx, dy, dz, dt, etc.

3. Allocate host memory for phi_host and u_host.

4. Initialize phi and u using initializationPhi and initializationU functions, which in CUDA will be kernels.

5. Allocate device memory for all necessary arrays: phi_old, phi_new, u_old, u_new, Fx, Fy, Fz.

6. Copy initial phi and u from host to device.

7. Enter the main loop for num_steps:

   a. Launch calculateForce kernel.

   b. Launch allenCahn kernel.

   c. Launch boundaryConditionsPhi kernel.

   d. Launch thermalEquation kernel.

   e. Launch boundaryConditionsU kernel.

   f. Swap pointers for phi and u.

8. After the loop, copy the results back to the host.

9. Verify the results if needed.

10. Free all allocated memory.

Now, considering the functions:

- calculateForce: Computes Fx, Fy, Fz based on gradients. Each thread computes one point.

- allenCahn: Computes phinew based on phiold, uold, and forces.

- boundaryConditionsPhi: Sets boundaries to -1.0.

- thermalEquation: Computes unew based on uold and phinew.

- boundaryConditionsU: Sets boundaries to -delta.

Each of these will be a kernel.

Now, let's think about the kernel launches. For each kernel, I need to specify the grid and block dimensions. For a 3D grid, the block size can be, for example, 8x8x8, and the grid size will be (DATAXSIZE / 8, DATAYSIZE / 8, DATAZSIZE / 8), but I need to ensure that it covers all points, so using ( (DATAXSIZE + block.x -1)/block.x, ... ) is better.

But in the original code, the loops are from 0 to DATAXSIZE-1, etc., so the kernel needs to handle all these points.

Now, considering the function parameters. In CUDA, passing large arrays as parameters is done via pointers. So, each function will take pointers to the necessary arrays.

For example, the calculateForce function will take pointers to phi, Fx, Fy, Fz, and the constants.

Now, let's think about the data types. The original code uses double, so in CUDA, I'll continue using double.

Another consideration is the use of shared memory for performance. For example, in calculateForce, each thread computes gradients, which involve accessing neighboring points. Using shared memory to cache these points can improve performance, but for simplicity, I might not implement it now.

Now, let's outline the code structure.

First, include necessary headers:

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Then, define the necessary constants:

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400
#define SQ(x) ((x)*(x))

typedef double nRarray[DATAYSIZE][DATAXSIZE];

Then, declare the device functions:

__device__ double dFphi(double phi, double u, double lambda) {
    return (-phi*(1.0 - phi*phi) + lambda*u*(1.0 - phi*phi)*(1.0 - phi*phi));
}

__device__ double GradientX(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
    return (phi[idx + DATAXSIZE] - phi[idx - DATAXSIZE]) / (2.0*dx);
}

Wait, but in the original code, phi is a 3D array, so in CUDA, it's a 1D array, and the index is calculated as x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z.

But wait, in the original code, phi is a 3D array: phi[x][y][z]. So, in CUDA, to represent this as a 1D array, the index would be x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z.

But wait, DATAXSIZE is the size in x, so for each x, there are DATAYSIZE*DATAXSIZE elements. So, the index is correct.

But in the original code, the functions like GradientX are using phi[x+1][y][z], which in 1D would be (x+1)*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z.

So, in the device functions, I need to calculate the index correctly.

Alternatively, perhaps it's better to pass the size parameters to the functions so that they can calculate the index.

But for simplicity, I'll assume that the functions know the sizes.

Now, the main functions:

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, double epsilon, double W0, double tau0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x > 0 && x < DATAXSIZE-1 && y > 0 && y < DATAYSIZE-1 && z > 0 && z < DATAZSIZE-1) {
            double phix = GradientX(phi, dx, dy, dz, x, y, z);
            double phiy = GradientY(phi, dx, dy, dz, x, y, z);
            double phiz = GradientZ(phi, dx, dy, dz, x, y, z);
            double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
            double c = 16.0 * W0 * epsilon;
            double w = Wn(phix, phiy, phiz, epsilon, W0);
            double w2 = SQ(w);

            int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
            Fx[idx] = w2 * phix + sqGphi * w * c * dFunc(phix, phiy, phiz);
            Fy[idx] = w2 * phiy + sqGphi * w * c * dFunc(phiy, phiz, phix);
            Fz[idx] = w2 * phiz + sqGphi * w * c * dFunc(phiz, phix, phiy);
        } else {
            int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
            Fx[idx] = 0.0;
            Fy[idx] = 0.0;
            Fz[idx] = 0.0;
        }
    }
}

Wait, but in the original code, the functions like GradientX are using the 3D array, so in CUDA, I need to pass the 1D array and calculate the index.

Alternatively, perhaps it's better to pass the size parameters to the functions so that they can calculate the index.

But for now, I'll proceed with the assumption that the functions know the sizes.

Now, considering the main function:

In the main function, after allocating device memory, I'll launch each kernel.

For example:

dim3 blockSize(8, 8, 8);
dim3 gridSize((DATAXSIZE + blockSize.x - 1)/blockSize.x, 
              (DATAYSIZE + blockSize.y - 1)/blockSize.y, 
              (DATAZSIZE + blockSize.z - 1)/blockSize.z);

calculateForce<<<gridSize, blockSize>>>(d_phiold, d_Fx, d_Fy, d_Fz, dx, dy, dz, epsilon, W0, tau0);
cudaCheckError();

But I need to ensure that the grid size is correctly calculated to cover all points.

Now, considering the swapGrid function. In CUDA, instead of copying data, I can swap pointers. So, I'll have two sets of pointers for phi and u.

For example:

double *phi_old = d_phiold;
double *phi_new = d_phinew;

After computing phi_new, I'll swap them:

double *temp = phi_old;
phi_old = phi_new;
phi_new = temp;

But I need to ensure that the pointers are correctly managed.

Now, considering the error checking. I'll need a helper function to check CUDA errors:

void cudaCheckError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

I'll call this after each CUDA API call.

Now, putting it all together, the main function will:

1. Read num_steps.

2. Initialize constants.

3. Allocate host memory.

4. Initialize phi and u using CUDA kernels.

5. Allocate device memory.

6. Copy initial data to device.

7. Enter the main loop, launching each kernel.

8. After the loop, copy results back to host.

9. Verify results.

10. Free memory.

Now, considering the initialization functions:

__global__ void initializationPhi(double *phi, double r0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        double rx = x - 0.5*DATAXSIZE;
        double ry = y - 0.5*DATAYSIZE;
        double rz = z - 0.5*DATAZSIZE;
        double r = sqrt(SQ(rx) + SQ(ry) + SQ(rz));
        int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
        phi[idx] = (r < r0) ? 1.0 : -1.0;
    }
}

Similarly for initializationU.

Now, considering the boundary conditions:

__global__ void boundaryConditionsPhi(double *phi) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x == 0 || x == DATAXSIZE-1 || y == 0 || y == DATAYSIZE-1 || z == 0 || z == DATAZSIZE-1) {
            int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
            phi[idx] = -1.0;
        }
    }
}

Similarly for boundaryConditionsU.

Now, considering the thermalEquation:

__global__ void thermalEquation(double *unew, double *uold, double *phinew, double *phiold, double D, double dt, double dx, double dy, double dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x > 0 && x < DATAXSIZE-1 && y > 0 && y < DATAYSIZE-1 && z > 0 && z < DATAZSIZE-1) {
            double laplacian = Laplacian(uold, dx, dy, dz, x, y, z);
            int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
            unew[idx] = uold[idx] + 0.5*(phinew[idx] - phiold[idx]) + dt * D * laplacian;
        }
    }
}

But I need to implement the Laplacian function as a device function.

__device__ double Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
    double phixx = (phi[idx + DATAXSIZE] + phi[idx - DATAXSIZE] - 2.0*phi[idx]) / (SQ(dx));
    double phiyy = (phi[idx + DATAYSIZE*DATAXSIZE] + phi[idx - DATAYSIZE*DATAXSIZE] - 2.0*phi[idx]) / (SQ(dy));
    double phizz = (phi[idx + DATAYSIZE*DATAXSIZE*DATAZSIZE] + phi[idx - DATAYSIZE*DATAXSIZE*DATAZSIZE] - 2.0*phi[idx]) / (SQ(dz));
    return phixx + phiyy + phizz;
}

Wait, but in the original code, the Laplacian is computed as the sum of second derivatives in x, y, z. So, in CUDA, I need to compute these second derivatives correctly.

Now, considering the allenCahn function:

__global__ void allenCahn(double *phinew, double *phiold, double *uold, double *Fx, double *Fy, double *Fz, double epsilon, double W0, double tau0, double lambda, double dt, double dx, double dy, double dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x > 0 && x < DATAXSIZE-1 && y > 0 && y < DATAYSIZE-1 && z > 0 && z < DATAZSIZE-1) {
            double phix = GradientX(phiold, dx, dy, dz, x, y, z);
            double phiy = GradientY(phiold, dx, dy, dz, x, y, z);
            double phiz = GradientZ(phiold, dx, dy, dz, x, y, z);
            double divergence = Divergence(Fx, Fy, Fz, dx, dy, dz, x, y, z);
            double dfphi = dFphi(phiold[idx], uold[idx], lambda);
            int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
            phinew[idx] = phiold[idx] + (dt / taun(phix, phiy, phiz, epsilon, tau0)) * (divergence - dfphi);
        }
    }
}

But I need to implement the Divergence function as a device function.

__device__ double Divergence(double *phix, double *phiy, double *phiz, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z;
    return GradientX(phix, dx, dy, dz, x, y, z) + 
           GradientY(phiy, dx, dy, dz, x, y, z) +
           GradientZ(phiz, dx, dy, dz, x, y, z);
}

Wait, but in the original code, Divergence is a function that takes phix, phiy, phiz arrays and computes their divergence. So, in CUDA, each thread computes the divergence at (x, y, z) by computing the gradients of each component.

But in the original code, phix, phiy, phiz are 3D arrays, so in CUDA, they are 1D arrays, and the index is calculated as x*DATAYSIZE*DATAXSIZE + y*DATAXSIZE + z.

Now, considering the functions like An, Wn, taun, dFunc, which are used in calculateForce and allenCahn. These need to be implemented as device functions.

For example:

__device__ double An(double phix, double phiy, double phiz, double epsilon) {
    if (phix != 0.0 || phiy != 0.0 || phiz != 0.0) {
        double numerator = 1.0 - 3.0 * epsilon;
        double denominator = 1.0 - 3.0 * epsilon;
        double term = (4.0 * epsilon) / denominator;
        double a = SQ(phix) * SQ(phix) + SQ(phiy) * SQ(phiy) + SQ(phiz) * SQ(phiz);
        double b = SQ(phix) + SQ(phiy) + SQ(phiz);
        double c = a / (b * b);
        return numerator * (1.0 + term * c);
    } else {
        return 1.0 - (5.0/3.0)*epsilon;
    }
}

Similarly for Wn, taun, dFunc.

Now, putting it all together, the code will have:

- Device functions for all the helper functions.

- Global functions for each of the main functions.

- Proper kernel launches with grid and block dimensions.

- Memory management with cudaMemcpy and device pointers.

- Error checking after each CUDA API call.

- Pointer swapping for swapGrid.

Now, considering the main function's structure, I'll need to manage the data transfer and kernel launches correctly.

Finally, after translating all functions, I'll ensure that the code compiles without errors and preserves the original algorithm's functionality.
</think>

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DATAXSIZE 400
#define DATAYSIZE 400
#define DATAZSIZE 400
#define SQ(x) ((x)*(x))

typedef double nRarray[DATAYSIZE][DATAXSIZE];

#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

__device__ double dFphi(double phi, double u, double lambda) {
    return (-phi * (1.0 - phi * phi) + lambda * u * (1.0 - phi * phi) * (1.0 - phi * phi));
}

__device__ double GradientX(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    return (phi[idx + DATAXSIZE] - phi[idx - DATAXSIZE]) / (2.0 * dx);
}

__device__ double GradientY(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    return (phi[idx + DATAYSIZE * DATAXSIZE] - phi[idx - DATAYSIZE * DATAXSIZE]) / (2.0 * dy);
}

__device__ double GradientZ(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    return (phi[idx + DATAYSIZE * DATAXSIZE * DATAZSIZE] - phi[idx - DATAYSIZE * DATAXSIZE * DATAZSIZE]) / (2.0 * dz);
}

__device__ double Divergence(double *phix, double *phiy, double *phiz, double dx, double dy, double dz, int x, int y, int z) {
    return GradientX(phix, dx, dy, dz, x, y, z) + GradientY(phiy, dx, dy, dz, x, y, z) + GradientZ(phiz, dx, dy, dz, x, y, z);
}

__device__ double Laplacian(double *phi, double dx, double dy, double dz, int x, int y, int z) {
    int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
    double phixx = (phi[idx + DATAXSIZE] + phi[idx - DATAXSIZE] - 2.0 * phi[idx]) / SQ(dx);
    double phiyy = (phi[idx + DATAYSIZE * DATAXSIZE] + phi[idx - DATAYSIZE * DATAXSIZE] - 2.0 * phi[idx]) / SQ(dy);
    double phizz = (phi[idx + DATAYSIZE * DATAXSIZE * DATAZSIZE] + phi[idx - DATAYSIZE * DATAXSIZE * DATAZSIZE] - 2.0 * phi[idx]) / SQ(dz);
    return phixx + phiyy + phizz;
}

__device__ double An(double phix, double phiy, double phiz, double epsilon) {
    if (phix != 0.0 || phiy != 0.0 || phiz != 0.0) {
        double numerator = 1.0 - 3.0 * epsilon;
        double denominator = 1.0 - 3.0 * epsilon;
        double term = (4.0 * epsilon) / denominator;
        double a = SQ(phix) * SQ(phix) + SQ(phiy) * SQ(phiy) + SQ(phiz) * SQ(phiz);
        double b = SQ(phix) + SQ(phiy) + SQ(phiz);
        double c = a / (b * b);
        return numerator * (1.0 + term * c);
    } else {
        return 1.0 - (5.0 / 3.0) * epsilon;
    }
}

__device__ double Wn(double phix, double phiy, double phiz, double epsilon, double W0) {
    return W0 * An(phix, phiy, phiz, epsilon);
}

__device__ double taun(double phix, double phiy, double phiz, double epsilon, double tau0) {
    return tau0 * SQ(An(phix, phiy, phiz, epsilon));
}

__device__ double dFunc(double l, double m, double n) {
    if (l != 0.0 || m != 0.0 || n != 0.0) {
        double numerator = (l * l * l * (SQ(m) + SQ(n)) - l * (SQ(m) * SQ(m) + SQ(n) * SQ(n)));
        double denominator = (SQ(l) + SQ(m) + SQ(n)) * (SQ(l) + SQ(m) + SQ(n));
        return numerator / denominator;
    } else {
        return 0.0;
    }
}

__global__ void calculateForce(double *phi, double *Fx, double *Fy, double *Fz, double dx, double dy, double dz, double epsilon, double W0, double tau0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x > 0 && x < DATAXSIZE - 1 && y > 0 && y < DATAYSIZE - 1 && z > 0 && z < DATAZSIZE - 1) {
            double phix = GradientX(phi, dx, dy, dz, x, y, z);
            double phiy = GradientY(phi, dx, dy, dz, x, y, z);
            double phiz = GradientZ(phi, dx, dy, dz, x, y, z);
            double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
            double c = 16.0 * W0 * epsilon;
            double w = Wn(phix, phiy, phiz, epsilon, W0);
            double w2 = SQ(w);

            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            Fx[idx] = w2 * phix + sqGphi * w * c * dFunc(phix, phiy, phiz);
            Fy[idx] = w2 * phiy + sqGphi * w * c * dFunc(phiy, phiz, phix);
            Fz[idx] = w2 * phiz + sqGphi * w * c * dFunc(phiz, phix, phiy);
        } else {
            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            Fx[idx] = 0.0;
            Fy[idx] = 0.0;
            Fz[idx] = 0.0;
        }
    }
}

__global__ void allenCahn(double *phinew, double *phiold, double *uold, double *Fx, double *Fy, double *Fz, double epsilon, double W0, double tau0, double lambda, double dt, double dx, double dy, double dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x > 0 && x < DATAXSIZE - 1 && y > 0 && y < DATAYSIZE - 1 && z > 0 && z < DATAZSIZE - 1) {
            double phix = GradientX(phiold, dx, dy, dz, x, y, z);
            double phiy = GradientY(phiold, dx, dy, dz, x, y, z);
            double phiz = GradientZ(phiold, dx, dy, dz, x, y, z);
            double divergence = Divergence(Fx, Fy, Fz, dx, dy, dz, x, y, z);
            double dfphi = dFphi(phiold[x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z], uold[x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z], lambda);
            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            phinew[idx] = phiold[idx] + (dt / taun(phix, phiy, phiz, epsilon, tau0)) * (divergence - dfphi);
        }
    }
}

__global__ void boundaryConditionsPhi(double *phi) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < DATAXSIZE && y < DATAYSIZE && z < DATAZSIZE) {
        if (x == 0 || x == DATAXSIZE - 1 || y == 0 || y == DATAYSIZE - 1 || z == 0 || z == DATAZSIZE - 1) {
            int idx = x * DATAYSIZE * DATAXSIZE + y * DATAXSIZE + z;
            phi[idx] = -1.0;
        }
    }
}

__global__ void thermalEquation(double *unew, double *uold, double *phinew, double *phiold, double D, double dt, double dx, double dy, double dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z