/*
* CUDA Lattice Boltzmann Method (LBM) Implementation for 2D Lid-Driven Cavity Flow
*
* Author: Aristotle Martin (CUDA port)
*
* This code implements a single-GPU CUDA version of the 2D lid-driven cavity flow
* simulation using the Lattice Boltzmann Method with a D2Q9 lattice structure.
* Based on the serial CPU implementation but optimized for GPU execution.
*
* CUDA IMPLEMENTATION STRATEGY:
* - Main computational kernels (collision+streaming, boundary conditions) run on GPU
* - Memory is allocated on GPU using standard cudaMalloc (not unified memory)
* - Explicit data transfers between host and device as needed
* - Thread blocks sized for optimal GPU occupancy
*
* Algorithm Overview:
* 1. Initialize fluid distributions to equilibrium values
* 2. For each time step:
*    a. GPU Collision+Streaming kernel: Relax distributions and move to neighbors
*    b. GPU Boundary condition kernel: Apply Zou-He BC for moving lid
*    c. Swap distribution arrays (ping-pong scheme)
* 3. Transfer results to host and output macroscopic quantities
*
* D2Q9 Lattice:
* - 9 velocity directions (including rest particle)
* - 2D square lattice with nearest and next-nearest neighbor connections
* - Velocity directions: (0,0), (±1,0), (0,±1), (±1,±1)
*/
#include <cuda.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <sys/time.h>

// Lattice Boltzmann Method constants
#define _STENCILSIZE_ 9    // Number of velocity directions in D2Q9 lattice
#define _LX_ 1024           // Grid size in x-direction (small size for testing)
#define _LY_ 1024           // Grid size in y-direction
#define _NDIMS_ 2          // Number of spatial dimensions
#define _INVALID_ -1       // Invalid grid index marker

// CUDA execution configuration
#define BLOCK_SIZE 256     // Threads per block (should be multiple of 32)

using namespace std;

// GPU constant memory for fast access to lattice parameters
__constant__ double omega_gpu;     // Relaxation parameter
__constant__ double uLid_gpu;      // Lid velocity
__constant__ int icx_gpu[_STENCILSIZE_];  // Lattice velocity x-components
__constant__ int icy_gpu[_STENCILSIZE_];  // Lattice velocity y-components
__constant__ double w_gpu[_STENCILSIZE_]; // Lattice weights
__constant__ int opp_gpu[_STENCILSIZE_];  // Opposite directions

/**
 * Device function: Convert 2D grid coordinates to linear array index
 *
 * @param i x-coordinate (0 to _LX_-1)
 * @param j y-coordinate (0 to _LY_-1)
 * @return Linear index for accessing arrays, or _INVALID_ if out of bounds
 */
__device__ int getGridIdx(int i, int j) {
    if (i < 0 || i >= _LX_ || j < 0 || j >= _LY_) {
        return _INVALID_;
    }
    return i + _LX_ * j;
}

/**
 * CUDA Kernel: Collision and Streaming Step
 *
 * Core computational kernel that performs both collision and streaming
 * operations for the entire fluid domain in parallel on the GPU.
 *
 * Each thread handles one lattice site and processes all 9 velocity directions.
 * The kernel combines collision (relaxation to equilibrium) and streaming
 * (movement to neighboring sites) for computational efficiency.
 *
 * @param distr Current distribution functions (input)
 * @param distrAdv New distribution functions after collision+streaming (output)
 * @param stencilOpPt Precomputed streaming destinations
 */
__global__ void collideStreamKernel(double* distr, double* distrAdv, int* stencilOpPt) {
    // Calculate global thread index (lattice site)
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: ensure thread corresponds to valid lattice site
    if (ii >= _LX_ * _LY_) return;

    // Calculate 2D coordinates from linear index
    int myI = ii % _LX_;   // x-coordinate
    int myJ = ii / _LX_;  // y-coordinate

    // STEP 1: Calculate macroscopic variables from distribution functions
    double rho = 0.0;  // Fluid density
    double ux = 0.0;   // x-momentum
    double uy = 0.0;   // y-momentum
    double distr_local[_STENCILSIZE_];

    // Load distributions and compute macroscopic quantities
    for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
        distr_local[iNbr] = distr[ii * _STENCILSIZE_ + iNbr];
        rho += distr_local[iNbr];                           // ρ = Σ f_i
        ux += distr_local[iNbr] * icx_gpu[iNbr];           // ρu_x = Σ f_i * c_ix
        uy += distr_local[iNbr] * icy_gpu[iNbr];           // ρu_y = Σ f_i * c_iy
    }

    // Convert momentum to velocity
    double orho = 1.0 / rho;
    ux *= orho;
    uy *= orho;
    double uke = ux * ux + uy * uy;  // Kinetic energy |u|²

    // STEP 2: Collision and Streaming
    for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
        // Streaming destination
        int nbrInd = stencilOpPt[ii * _STENCILSIZE_ + iNbr];

        // Lattice velocity dot product with fluid velocity
        double cdotu = icx_gpu[iNbr] * ux + icy_gpu[iNbr] * uy;

        // Equilibrium distribution (Maxwell-Boltzmann, 2nd order)
        double distr_eq = w_gpu[iNbr] * rho * (1.0 + 3.0*cdotu + 4.5*cdotu*cdotu - 1.5*uke);

        // BGK collision + streaming
        distrAdv[nbrInd] = omega_gpu * distr_eq + (1.0 - omega_gpu) * distr_local[iNbr];
    }
}

/**
 * CUDA Kernel: Zou-He Boundary Condition for Moving Lid
 *
 * Applies velocity boundary condition on the top wall (moving lid) using
 * the Zou-He method. Each thread handles one lattice site along the top boundary.
 *
 * The Zou-He method extrapolates density from the bulk fluid and reconstructs
 * unknown distribution functions to satisfy the prescribed velocity.
 *
 * @param distr Distribution functions to modify
 */
__global__ void zouHeBCKernel(double* distr) {
    // Each thread handles one x-coordinate along the top boundary
    int myI = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (myI >= _LX_) return;

    int myJ = 0; // Top boundary (y = 0)
    int idxIJ = getGridIdx(myI, myJ);

    // Prescribed boundary velocities
    double ux = uLid_gpu;  // Horizontal lid velocity
    double uy = 0.0;       // No vertical velocity

    // Extrapolate density using Zou-He formula
    double rho = (1.0/(1.0-uy)) * (distr[idxIJ*_STENCILSIZE_+0] +
                                   distr[idxIJ*_STENCILSIZE_+1] +
                                   distr[idxIJ*_STENCILSIZE_+3] +
                                   2*(distr[idxIJ*_STENCILSIZE_+4] +
                                      distr[idxIJ*_STENCILSIZE_+7] +
                                      distr[idxIJ*_STENCILSIZE_+8]));

    // Reconstruct unknown distributions
    distr[idxIJ*_STENCILSIZE_+2] = distr[idxIJ*_STENCILSIZE_+4] + (2.0/3.0)*rho*uy;

    distr[idxIJ*_STENCILSIZE_+5] = distr[idxIJ*_STENCILSIZE_+7] -
                                   (1.0/2.0)*(distr[idxIJ*_STENCILSIZE_+1] - distr[idxIJ*_STENCILSIZE_+3]) +
                                   (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;

    distr[idxIJ*_STENCILSIZE_+6] = distr[idxIJ*_STENCILSIZE_+8] +
                                   (1.0/2.0)*(distr[idxIJ*_STENCILSIZE_+1] - distr[idxIJ*_STENCILSIZE_+3]) -
                                   (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
}

/**
 * Host function: Write simulation results to output file
 *
 * Transfers final results from GPU to CPU and calculates macroscopic
 * quantities for output. Results written in CSV format.
 */
void writeOutput(double* distr_gpu, int* icx, int* icy) {
    // Allocate host memory for results
    double* distr_host = new double[_LX_ * _LY_ * _STENCILSIZE_];

    // Transfer results from GPU to CPU
    cudaMemcpy(distr_host, distr_gpu, _LX_ * _LY_ * _STENCILSIZE_ * sizeof(double), cudaMemcpyDeviceToHost);

    // Open output file
    std::ofstream out_file("out.txt");

    // Process each lattice site
    for (int idxI = 0; idxI < _LX_; idxI++) {
        for (int idxJ = 0; idxJ < _LY_; idxJ++) {
            int idxIJ = idxI + _LX_ * idxJ;

            // Calculate macroscopic quantities
            double rho = 0.0;
            double ux = 0.0;
            double uy = 0.0;

            for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
                double f = distr_host[idxIJ * _STENCILSIZE_ + iNbr];
                rho += f;
                ux += f * icx[iNbr];
                uy += f * icy[iNbr];
            }

            // Convert to velocity
            ux /= rho;
            uy /= rho;

            // Write to file
            out_file << std::setprecision(16) << idxI << ", " << idxJ << ", "
                     << ux << ", " << uy << ", " << rho << std::endl;
        }
    }

    out_file.close();
    delete[] distr_host;
}

/**
 * Host function: Setup streaming adjacency table
 *
 * Precomputes where each distribution function will stream to,
 * implementing both fluid streaming and bounce-back boundary conditions.
 */
void setupAdjacency(int* stencilOpPt, int* icx, int* icy, int* opp) {
    for (int ii = 0; ii < _LX_ * _LY_; ii++) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_;

        for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
            int nbrI = myI + icx[iNbr];
            int nbrJ = myJ + icy[iNbr];

            // Check bounds
            if (nbrI < 0 || nbrI >= _LX_ || nbrJ < 0 || nbrJ >= _LY_) {
                // Bounce-back boundary condition
                stencilOpPt[ii * _STENCILSIZE_ + iNbr] = ii * _STENCILSIZE_ + opp[iNbr];
            } else {
                // Normal streaming
                int nbrIJ = nbrI + _LX_ * nbrJ;
                stencilOpPt[ii * _STENCILSIZE_ + iNbr] = nbrIJ * _STENCILSIZE_ + iNbr;
            }
        }
    }
}

/**
 * Host function: Initialize fluid to equilibrium state
 */
void initializeFluid(double* distr_gpu, double* w) {
    // Create host array for initialization
    double* distr_host = new double[_LX_ * _LY_ * _STENCILSIZE_];

    // Initialize to equilibrium (fluid at rest, unit density)
    for (int ii = 0; ii < _LX_ * _LY_; ii++) {
        for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
            distr_host[ii * _STENCILSIZE_ + iNbr] = w[iNbr];
        }
    }

    // Transfer to GPU
    cudaMemcpy(distr_gpu, distr_host, _LX_ * _LY_ * _STENCILSIZE_ * sizeof(double), cudaMemcpyHostToDevice);

    delete[] distr_host;
}

/**
 * MAIN PROGRAM: Single-GPU CUDA Cavity Flow Simulation
 */
int main() {
    // =============================================================================
    // SIMULATION PARAMETERS
    // =============================================================================
    int maxT = 100;        // Total time steps
    double uLid = 0.05;      // Lid velocity
    double Re = 100.0;       // Reynolds number

    // LBM parameters
    double cs2 = 1.0/3.0;
    double nu = uLid * _LX_ / Re;
    double omega = 1.0 / (3.0*nu + 0.5);

    // =============================================================================
    // D2Q9 LATTICE CONSTANTS
    // =============================================================================
    int icx[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};
    int icy[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1};
    int opp[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};
    double w[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,
                              1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};

    // =============================================================================
    // GPU MEMORY ALLOCATION
    // =============================================================================
    cout << "Initializing CUDA Cavity Flow Simulation..." << endl;
    cout << "Grid size: " << _LX_ << "x" << _LY_ << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Lid velocity: " << uLid << endl;
    cout << "Relaxation parameter: " << omega << endl;
    cout << "Time steps: " << maxT << endl << endl;

    // GPU memory allocation
    double* distr_gpu;
    double* distrAdv_gpu;
    int* stencilOpPt_gpu;

    size_t distr_size = _LX_ * _LY_ * _STENCILSIZE_ * sizeof(double);
    size_t stencil_size = _LX_ * _LY_ * _STENCILSIZE_ * sizeof(int);

    cudaMalloc(&distr_gpu, distr_size);
    cudaMalloc(&distrAdv_gpu, distr_size);
    cudaMalloc(&stencilOpPt_gpu, stencil_size);

    // Copy constants to GPU constant memory
    cudaMemcpyToSymbol(omega_gpu, &omega, sizeof(double));
    cudaMemcpyToSymbol(uLid_gpu, &uLid, sizeof(double));
    cudaMemcpyToSymbol(icx_gpu, icx, _STENCILSIZE_ * sizeof(int));
    cudaMemcpyToSymbol(icy_gpu, icy, _STENCILSIZE_ * sizeof(int));
    cudaMemcpyToSymbol(w_gpu, w, _STENCILSIZE_ * sizeof(double));
    cudaMemcpyToSymbol(opp_gpu, opp, _STENCILSIZE_ * sizeof(int));

    // =============================================================================
    // INITIALIZATION
    // =============================================================================
    // Setup streaming adjacency on host
    int* stencilOpPt_host = new int[_LX_ * _LY_ * _STENCILSIZE_];
    setupAdjacency(stencilOpPt_host, icx, icy, opp);
    cudaMemcpy(stencilOpPt_gpu, stencilOpPt_host, stencil_size, cudaMemcpyHostToDevice);

    // Initialize fluid distributions
    initializeFluid(distr_gpu, w);

    // CUDA execution configuration
    int numSites = _LX_ * _LY_;
    int numBlocks = (numSites + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int numBlocksBC = (_LX_ + BLOCK_SIZE - 1) / BLOCK_SIZE;  // For boundary condition

    cout << "CUDA Configuration:" << endl;
    cout << "Threads per block: " << BLOCK_SIZE << endl;
    cout << "Blocks for collision: " << numBlocks << endl;
    cout << "Blocks for boundary: " << numBlocksBC << endl << endl;

    // =============================================================================
    // MAIN SIMULATION LOOP
    // =============================================================================
    cout << "Starting CUDA simulation..." << endl;

    // Timing
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int t = 0; t < maxT; t++) {
        // Step 1: Collision + Streaming kernel
        collideStreamKernel<<<numBlocks, BLOCK_SIZE>>>(distr_gpu, distrAdv_gpu, stencilOpPt_gpu);

        // Step 2: Boundary condition kernel
        zouHeBCKernel<<<numBlocksBC, BLOCK_SIZE>>>(distrAdv_gpu);

        // Step 3: Swap arrays (ping-pong)
        double* temp = distr_gpu;
        distr_gpu = distrAdv_gpu;
        distrAdv_gpu = temp;

        // Progress output
        if (t % 1000 == 0) {
            cout << "Completed " << t << " / " << maxT << " time steps" << endl;
        }
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    long total_updates = (long)_LX_ * _LY_ * maxT;
    double mflups = total_updates / elapsed_time / 1000000.0;

    cout << "Simulation completed!" << endl;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    cout << "Performance: " << mflups << " MFLUPS" << endl;

    // =============================================================================
    // OUTPUT RESULTS
    // =============================================================================
    cout << "Writing results to out.txt..." << endl;
    writeOutput(distr_gpu, icx, icy);
    cout << "Done!" << endl;

    // =============================================================================
    // CLEANUP
    // =============================================================================
    cudaFree(distr_gpu);
    cudaFree(distrAdv_gpu);
    cudaFree(stencilOpPt_gpu);
    delete[] stencilOpPt_host;

    return 0;
}
