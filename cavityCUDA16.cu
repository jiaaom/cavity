/*
* CUDA Lattice Boltzmann Method (LBM) Implementation for 2D Lid-Driven Cavity Flow
*
* Author: Aristotle Martin (CUDA port)
*
* This code implements a single-GPU CUDA version of the 2D lid-driven cavity flow
* simulation using the Lattice Boltzmann Method with a D2Q9 lattice structure.
* Based on the serial CPU implementation but optimized for GPU execution.
*
* This variant stores lattice distributions in FP16 to explore memory-bandwidth
* and precision trade-offs relative to the FP64 baseline.
*/
#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

// Lattice Boltzmann Method constants
#define _STENCILSIZE_ 9    // Number of velocity directions in D2Q9 lattice
#define _LX_ 1024          // Grid size in x-direction (small size for testing)
#define _LY_ 1024          // Grid size in y-direction
#define _NDIMS_ 2          // Number of spatial dimensions
#define _INVALID_ -1       // Invalid grid index marker

// CUDA execution configuration
#define BLOCK_SIZE 256     // Threads per block (should be multiple of 32)

using namespace std;

// GPU constant memory for fast access to lattice parameters
__constant__ __half omega_gpu;                     // Relaxation parameter
__constant__ __half uLid_gpu;                      // Lid velocity
__constant__ int icx_gpu[_STENCILSIZE_];           // Lattice velocity x-components
__constant__ int icy_gpu[_STENCILSIZE_];           // Lattice velocity y-components
__constant__ __half w_gpu[_STENCILSIZE_];          // Lattice weights
__constant__ int opp_gpu[_STENCILSIZE_];           // Opposite directions

/**
 * Device function: Convert 2D grid coordinates to linear array index.
 */
__device__ int getGridIdx(int i, int j) {
    if (i < 0 || i >= _LX_ || j < 0 || j >= _LY_) {
        return _INVALID_;
    }
    return i + _LX_ * j;
}

/**
 * CUDA Kernel: Collision and Streaming Step using FP16 storage.
 */
__global__ void collideStreamKernel(const __half* distr, __half* distrAdv, const int* stencilOpPt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii >= _LX_ * _LY_) {
        return;
    }

    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;
    __half distrLocal[_STENCILSIZE_];

    for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
        __half fHalf = distr[ii * _STENCILSIZE_ + iNbr];
        distrLocal[iNbr] = fHalf;
        float f = __half2float(fHalf);
        rho += f;
        ux += f * static_cast<float>(icx_gpu[iNbr]);
        uy += f * static_cast<float>(icy_gpu[iNbr]);
    }

    float invRho = rho > 0.0f ? 1.0f / rho : 0.0f;
    ux *= invRho;
    uy *= invRho;
    float uke = ux * ux + uy * uy;
    float omega = __half2float(omega_gpu);

    for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
        int nbrInd = stencilOpPt[ii * _STENCILSIZE_ + iNbr];
        if (nbrInd < 0) {
            continue;
        }

        float cdotu = static_cast<float>(icx_gpu[iNbr]) * ux + static_cast<float>(icy_gpu[iNbr]) * uy;
        float weight = __half2float(w_gpu[iNbr]);
        float distrEq = weight * rho * (1.0f + 3.0f * cdotu + 4.5f * cdotu * cdotu - 1.5f * uke);
        float distrOld = __half2float(distrLocal[iNbr]);
        float distrNew = omega * distrEq + (1.0f - omega) * distrOld;
        distrAdv[nbrInd] = __float2half_rn(distrNew);
    }
}

/**
 * CUDA Kernel: Zou-He Boundary Condition for Moving Lid.
 */
__global__ void zouHeBCKernel(__half* distr) {
    int myI = blockIdx.x * blockDim.x + threadIdx.x;
    if (myI >= _LX_) {
        return;
    }

    int myJ = 0;
    int idxIJ = getGridIdx(myI, myJ);
    if (idxIJ == _INVALID_) {
        return;
    }

    float ux = __half2float(uLid_gpu);
    float uy = 0.0f;

    float f0 = __half2float(distr[idxIJ * _STENCILSIZE_ + 0]);
    float f1 = __half2float(distr[idxIJ * _STENCILSIZE_ + 1]);
    float f2 = __half2float(distr[idxIJ * _STENCILSIZE_ + 2]);
    float f3 = __half2float(distr[idxIJ * _STENCILSIZE_ + 3]);
    float f4 = __half2float(distr[idxIJ * _STENCILSIZE_ + 4]);
    float f5 = __half2float(distr[idxIJ * _STENCILSIZE_ + 5]);
    float f6 = __half2float(distr[idxIJ * _STENCILSIZE_ + 6]);
    float f7 = __half2float(distr[idxIJ * _STENCILSIZE_ + 7]);
    float f8 = __half2float(distr[idxIJ * _STENCILSIZE_ + 8]);

    float rho = (1.0f / (1.0f - uy)) * (f0 + f1 + f3 + 2.0f * (f4 + f7 + f8));

    float f2New = f4 + (2.0f / 3.0f) * rho * uy;
    float f5New = f7 - 0.5f * (f1 - f3) + 0.5f * rho * ux - (1.0f / 6.0f) * rho * uy;
    float f6New = f8 + 0.5f * (f1 - f3) - 0.5f * rho * ux - (1.0f / 6.0f) * rho * uy;

    distr[idxIJ * _STENCILSIZE_ + 2] = __float2half_rn(f2New);
    distr[idxIJ * _STENCILSIZE_ + 5] = __float2half_rn(f5New);
    distr[idxIJ * _STENCILSIZE_ + 6] = __float2half_rn(f6New);
}

/**
 * Host function: Write simulation results to output file.
 */
void writeOutput(const __half* distr_gpu, const int* icx, const int* icy) {
    size_t totalSites = static_cast<size_t>(_LX_) * _LY_;
    size_t dataSize = totalSites * _STENCILSIZE_;
    __half* distr_host = new __half[dataSize];

    cudaMemcpy(distr_host, distr_gpu, dataSize * sizeof(__half), cudaMemcpyDeviceToHost);

    ofstream out_file("out.txt");
    for (int idxI = 0; idxI < _LX_; idxI++) {
        for (int idxJ = 0; idxJ < _LY_; idxJ++) {
            int idxIJ = idxI + _LX_ * idxJ;

            double rho = 0.0;
            double ux = 0.0;
            double uy = 0.0;

            for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
                double f = static_cast<double>(__half2float(distr_host[idxIJ * _STENCILSIZE_ + iNbr]));
                rho += f;
                ux += f * icx[iNbr];
                uy += f * icy[iNbr];
            }

            ux /= rho;
            uy /= rho;

            out_file << setprecision(16) << idxI << ", " << idxJ << ", "
                     << ux << ", " << uy << ", " << rho << endl;
        }
    }

    out_file.close();
    delete[] distr_host;
}

/**
 * Host function: Setup streaming adjacency table.
 */
void setupAdjacency(int* stencilOpPt, const int* icx, const int* icy, const int* opp) {
    for (int ii = 0; ii < _LX_ * _LY_; ii++) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_;

        for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
            int nbrI = myI + icx[iNbr];
            int nbrJ = myJ + icy[iNbr];

            if (nbrI < 0 || nbrI >= _LX_ || nbrJ < 0 || nbrJ >= _LY_) {
                stencilOpPt[ii * _STENCILSIZE_ + iNbr] = ii * _STENCILSIZE_ + opp[iNbr];
            } else {
                int nbrIJ = nbrI + _LX_ * nbrJ;
                stencilOpPt[ii * _STENCILSIZE_ + iNbr] = nbrIJ * _STENCILSIZE_ + iNbr;
            }
        }
    }
}

/**
 * Host function: Initialize fluid to equilibrium state.
 */
void initializeFluid(__half* distr_gpu, const __half* w) {
    size_t totalSites = static_cast<size_t>(_LX_) * _LY_;
    size_t dataSize = totalSites * _STENCILSIZE_;
    __half* distr_host = new __half[dataSize];

    for (size_t ii = 0; ii < totalSites; ii++) {
        for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
            distr_host[ii * _STENCILSIZE_ + iNbr] = w[iNbr];
        }
    }

    cudaMemcpy(distr_gpu, distr_host, dataSize * sizeof(__half), cudaMemcpyHostToDevice);
    delete[] distr_host;
}

/**
 * MAIN PROGRAM: Single-GPU CUDA Cavity Flow Simulation (FP16 storage).
 */
int main() {
    int maxT = 100;
    double uLid = 0.05;
    double Re = 100.0;

    double cs2 = 1.0 / 3.0;
    double nu = uLid * _LX_ / Re;
    double omega = 1.0 / (3.0 * nu + 0.5);

    int icx[_STENCILSIZE_] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    int icy[_STENCILSIZE_] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    int opp[_STENCILSIZE_] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    double wDouble[_STENCILSIZE_] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                                     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    cout << "Initializing CUDA FP16 Cavity Flow Simulation..." << endl;
    cout << "Grid size: " << _LX_ << "x" << _LY_ << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Lid velocity: " << uLid << endl;
    cout << "Relaxation parameter: " << omega << endl;
    cout << "Time steps: " << maxT << endl << endl;

    __half omegaHalf = __float2half_rn(static_cast<float>(omega));
    __half uLidHalf = __float2half_rn(static_cast<float>(uLid));
    array<__half, _STENCILSIZE_> wHalf;
    for (int i = 0; i < _STENCILSIZE_; i++) {
        wHalf[i] = __float2half_rn(static_cast<float>(wDouble[i]));
    }

    __half* distr_gpu = nullptr;
    __half* distrAdv_gpu = nullptr;
    int* stencilOpPt_gpu = nullptr;

    size_t distrSize = static_cast<size_t>(_LX_) * _LY_ * _STENCILSIZE_ * sizeof(__half);
    size_t stencilSize = static_cast<size_t>(_LX_) * _LY_ * _STENCILSIZE_ * sizeof(int);

    cudaMalloc(&distr_gpu, distrSize);
    cudaMalloc(&distrAdv_gpu, distrSize);
    cudaMalloc(&stencilOpPt_gpu, stencilSize);

    cudaMemcpyToSymbol(omega_gpu, &omegaHalf, sizeof(__half));
    cudaMemcpyToSymbol(uLid_gpu, &uLidHalf, sizeof(__half));
    cudaMemcpyToSymbol(icx_gpu, icx, _STENCILSIZE_ * sizeof(int));
    cudaMemcpyToSymbol(icy_gpu, icy, _STENCILSIZE_ * sizeof(int));
    cudaMemcpyToSymbol(w_gpu, wHalf.data(), _STENCILSIZE_ * sizeof(__half));
    cudaMemcpyToSymbol(opp_gpu, opp, _STENCILSIZE_ * sizeof(int));

    int* stencilOpPt_host = new int[_LX_ * _LY_ * _STENCILSIZE_];
    setupAdjacency(stencilOpPt_host, icx, icy, opp);
    cudaMemcpy(stencilOpPt_gpu, stencilOpPt_host, stencilSize, cudaMemcpyHostToDevice);

    initializeFluid(distr_gpu, wHalf.data());

    int numSites = _LX_ * _LY_;
    int numBlocks = (numSites + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksBC = (_LX_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cout << "CUDA Configuration:" << endl;
    cout << "Threads per block: " << BLOCK_SIZE << endl;
    cout << "Blocks for collision: " << numBlocks << endl;
    cout << "Blocks for boundary: " << numBlocksBC << endl << endl;

    cout << "Starting CUDA FP16 simulation..." << endl;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, nullptr);

    for (int t = 0; t < maxT; t++) {
        collideStreamKernel<<<numBlocks, BLOCK_SIZE>>>(distr_gpu, distrAdv_gpu, stencilOpPt_gpu);
        zouHeBCKernel<<<numBlocksBC, BLOCK_SIZE>>>(distrAdv_gpu);

        __half* temp = distr_gpu;
        distr_gpu = distrAdv_gpu;
        distrAdv_gpu = temp;

        if (t % 1000 == 0) {
            cout << "Completed " << t << " / " << maxT << " time steps" << endl;
        }
    }

    cudaDeviceSynchronize();

    gettimeofday(&end_time, nullptr);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    long total_updates = static_cast<long>(_LX_) * _LY_ * maxT;
    double mflups = total_updates / elapsed_time / 1000000.0;

    cout << "Simulation completed!" << endl;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    cout << "Performance: " << mflups << " MFLUPS" << endl;

    cout << "Writing results to out.txt..." << endl;
    writeOutput(distr_gpu, icx, icy);
    cout << "Done!" << endl;

    cudaFree(distr_gpu);
    cudaFree(distrAdv_gpu);
    cudaFree(stencilOpPt_gpu);
    delete[] stencilOpPt_host;

    return 0;
}
