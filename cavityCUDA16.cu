// CUDA LBM D2Q9, FP16 storage, SoA layout, pull-streaming, fixed Zou–He top lid
// Minimal, clear, benchmarkable.

#include <cuda.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/time.h>

#ifndef LX
#define LX 1024
#endif
#ifndef LY
#define LY 1024
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

constexpr int Q = 9;
__device__ __constant__ int cx_c[Q]   = {0, 1, 0,-1, 0, 1,-1,-1, 1};
__device__ __constant__ int cy_c[Q]   = {0, 0, 1, 0,-1, 1, 1,-1,-1};
__device__ __constant__ int opp_c[Q]  = {0, 3, 4, 1, 2, 7, 8, 5, 6};
__device__ __constant__ __half w_c[Q];
__device__ __constant__ __half omega_c;
__device__ __constant__ __half uLid_c;

// Utility
__host__ __device__ inline int idx2d(int x, int y) { return x + y * LX; }

// Kernel 1: collide in-place from f_in -> f_post (both SoA). FP32 math, FP16 storage.
// f[q] planes are sized N = LX*LY
__global__ void collide_kernel(const __half* __restrict__ f_in[Q], __half* __restrict__ f_post[Q]) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = LX * LY;
    if (gid >= N) return;

    // Load populations for this node
    float f[Q];
    #pragma unroll
    for (int q = 0; q < Q; ++q) f[q] = __half2float(f_in[q][gid]);

    // Macros
    float rho = 0.f, ux = 0.f, uy = 0.f;
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        rho += f[q];
        ux  += f[q] * (float)cx_c[q];
        uy  += f[q] * (float)cy_c[q];
    }
    float inv_rho = rho > 0.f ? 1.f / rho : 0.f;
    ux *= inv_rho; uy *= inv_rho;

    const float uke = ux*ux + uy*uy;
    const float omega = __half2float(omega_c);

    // BGK collide
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float cu = 3.f * (cx_c[q]*ux + cy_c[q]*uy);
        const float cu2 = 0.5f * cu * cu;          // 9/2 * (e·u)^2 with 3 factored in above -> 0.5*(3e·u)^2
        const float uu = 1.5f * uke;               // 3/2 |u|^2
        const float w = __half2float(w_c[q]);
        const float feq = w * rho * (1.f + cu + cu2 - uu);
        const float fout = f[q] + omega * (feq - f[q]);
        f_post[q][gid] = __float2half_rn(fout);
    }
}

// Kernel 2: pull streaming with on-the-fly neighbor index and walls.
// Writes into f_out. Handles bounce-back on left/right/bottom and Zou–He moving lid on top (y=LY-1).
__global__ void stream_bc_kernel(const __half* __restrict__ f_post[Q], __half* __restrict__ f_out[Q]) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = LX * LY;
    if (gid >= N) return;

    const int x = gid % LX;
    const int y = gid / LX;

    // For each direction q, we pull from neighbor at (x - cx[q], y - cy[q])
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        const int xn = x - cx_c[q];
        const int yn = y - cy_c[q];
        __half val;

        if (xn >= 0 && xn < LX && yn >= 0 && yn < LY) {
            val = f_post[q][idx2d(xn, yn)];
        } else {
            // Boundary handling: which boundary are we crossing?
            const int qo = opp_c[q];

            // Top moving lid at y = LY-1. We cross top if yn == LY.
            if (yn == LY) {
                // Zou–He moving lid, ux = uLid, uy = 0
                const float uLid = __half2float(uLid_c);

                // We reconstruct the incoming directions at top: q in {2,5,6} pull from outside.
                // Formulas (standard Zou–He):
                // f2 = f4 + (2/3) rho uy  with uy=0 -> f2 = f4
                // f5 = f7 + (1/2)(f1 - f3) + (1/6)rho(2ux - uy) -> with uy=0 -> f5 = f7 + 0.5(f1 - f3) + (1/3)rho ux
                // f6 = f8 + (1/2)(f3 - f1) + (1/6)rho(-2ux - uy) -> with uy=0 -> f6 = f8 + 0.5(f3 - f1) - (1/3)rho ux
                // We need rho at boundary node. Approximate from known post-collision populations at the node.

                // Load known post-collision values at boundary node (current cell)
                float f0 = __half2float(f_post[0][gid]);
                float f1 = __half2float(f_post[1][gid]);
                float f3 = __half2float(f_post[3][gid]);
                float f4 = __half2float(f_post[4][gid]);
                float f7 = __half2float(f_post[7][gid]);
                float f8 = __half2float(f_post[8][gid]);

                // Zou–He density closure with uy=0 at top:
                // rho = f0 + f1 + f3 + 2*(f4 + f7 + f8)  (standard lid BC)
                const float rho = f0 + f1 + f3 + 2.f * (f4 + f7 + f8);

                float f2 = f4;                       // uy=0
                float f5 = f7 + 0.5f*(f1 - f3) + (1.f/3.f)*rho * uLid;
                float f6 = f8 + 0.5f*(f3 - f1) - (1.f/3.f)*rho * uLid;

                if (q == 2)      val = __float2half_rn(f2);
                else if (q == 5) val = __float2half_rn(f5);
                else if (q == 6) val = __float2half_rn(f6);
                else              val = f_post[q][gid]; // not expected, but keep safe
            }
            // Bottom wall y = 0, cross bottom if yn == -1
            else if (yn == -1) {
                val = f_post[qo][gid]; // bounce-back
            }
            // Left wall x = 0, cross left if xn == -1
            else if (xn == -1) {
                val = f_post[qo][gid];
            }
            // Right wall x = LX-1, cross right if xn == LX
            else if (xn == LX) {
                val = f_post[qo][gid];
            }
            else {
                // Should not happen
                val = f_post[qo][gid];
            }
        }
        f_out[q][gid] = val;
    }
}

// Dump macroscopic fields to CSV (host-side accumulation)
static void write_output(const std::vector<__half*>& f_dev, const int LXh, const int LYh) {
    const int N = LXh * LYh;
    std::vector<__half> h(Q * N);
    for (int q = 0; q < Q; ++q) cudaMemcpy(h.data() + q*N, f_dev[q], N*sizeof(__half), cudaMemcpyDeviceToHost);

    std::ofstream out("out.csv");
    out << std::setprecision(16);
    for (int y = 0; y < LYh; ++y) {
        for (int x = 0; x < LXh; ++x) {
            int i = idx2d(x,y);
            double rho = 0.0, ux = 0.0, uy = 0.0;
            for (int q = 0; q < Q; ++q) {
                double f = (double)__half2float(h[q*N + i]);
                rho += f;
                ux  += f * cx_c[q];
                uy  += f * cy_c[q];
            }
            ux /= rho; uy /= rho;
            out << x << ',' << y << ',' << ux << ',' << uy << ',' << rho << '\n';
        }
    }
    out.close();
}

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

int main() {
    const int N = LX * LY;

    // Host weights
    const float w_h[Q] = {4.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,
                          1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f};

    // Params
    const double uLid = 0.05;
    const double Re   = 100.0;
    const double nu   = uLid * LX / Re;
    const double omega = 1.0 / (3.0*nu + 0.5);

    __half w_d[Q];
    for (int q = 0; q < Q; ++q) w_d[q] = __float2half_rn(w_h[q]);
    checkCuda(cudaMemcpyToSymbol(w_c, w_d, sizeof(w_d)), "cpy w");
    __half omega_h = __float2half_rn((float)omega);
    __half uLid_h  = __float2half_rn((float)uLid);
    checkCuda(cudaMemcpyToSymbol(omega_c, &omega_h, sizeof(__half)), "cpy omega");
    checkCuda(cudaMemcpyToSymbol(uLid_c, &uLid_h, sizeof(__half)), "cpy uLid");

    // Allocate SoA planes
    std::vector<__half*> f0(Q), f1(Q), f2(Q); // ping-pong buffers (in, post, out)
    for (int q = 0; q < Q; ++q) {
        checkCuda(cudaMalloc(&f0[q], N*sizeof(__half)), "malloc f0");
        checkCuda(cudaMalloc(&f1[q], N*sizeof(__half)), "malloc f1");
        checkCuda(cudaMalloc(&f2[q], N*sizeof(__half)), "malloc f2");
    }

    // Init to equilibrium at rest: f = w
    {
        std::vector<__half> tmp(N);
        for (int q = 0; q < Q; ++q) {
            std::fill(tmp.begin(), tmp.end(), w_d[q]);
            checkCuda(cudaMemcpy(f0[q], tmp.data(), N*sizeof(__half), cudaMemcpyHostToDevice), "init f0");
        }
    }

    const int threads = BLOCK_SIZE;
    const int blocks = (N + threads - 1) / threads;

    // Prepare device arrays of plane pointers (so kernels can index f[q])
    __half** f0_dev; __half** f1_dev; __half** f2_dev;
    checkCuda(cudaMalloc(&f0_dev, Q*sizeof(__half*)), "malloc f0_dev");
    checkCuda(cudaMalloc(&f1_dev, Q*sizeof(__half*)), "malloc f1_dev");
    checkCuda(cudaMalloc(&f2_dev, Q*sizeof(__half*)), "malloc f2_dev");

    checkCuda(cudaMemcpy(f0_dev, f0.data(), Q*sizeof(__half*), cudaMemcpyHostToDevice), "cpy f0_dev");
    checkCuda(cudaMemcpy(f1_dev, f1.data(), Q*sizeof(__half*), cudaMemcpyHostToDevice), "cpy f1_dev");
    checkCuda(cudaMemcpy(f2_dev, f2.data(), Q*sizeof(__half*), cudaMemcpyHostToDevice), "cpy f2_dev");

    // Time stepping
    const int maxT = 2; // set real steps for performance measurement
    // Timing
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int t = 0; t < maxT; ++t) {
        // collide: f0 -> f1
        collide_kernel<<<blocks, threads>>>((const __half** )f0_dev, (__half**)f1_dev);
        // stream+BC: f1 -> f2
        stream_bc_kernel<<<blocks, threads>>>((const __half**)f1_dev, (__half**)f2_dev);
        checkCuda(cudaGetLastError(), "kernels");

        // Rotate buffers: f2 becomes next f0
        std::swap(f0, f2);
        checkCuda(cudaMemcpy(f0_dev, f0.data(), Q*sizeof(__half*), cudaMemcpyHostToDevice), "swap f0_dev");
        checkCuda(cudaMemcpy(f2_dev, f2.data(), Q*sizeof(__half*), cudaMemcpyHostToDevice), "swap f2_dev");
    }
    checkCuda(cudaDeviceSynchronize(), "sync");

    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    long total_updates = (long)LX * LY * maxT;
    double mflups = total_updates / elapsed_time / 1000000.0;

    std::cout << "Simulation completed!" << std::endl;
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
    std::cout << "Performance: " << mflups << " MFLUPS" << std::endl;

    // Output
    write_output(f0, LX, LY);

    // Cleanup
    for (int q = 0; q < Q; ++q) { cudaFree(f0[q]); cudaFree(f1[q]); cudaFree(f2[q]); }
    cudaFree(f0_dev); cudaFree(f1_dev); cudaFree(f2_dev);

    return 0;
}
