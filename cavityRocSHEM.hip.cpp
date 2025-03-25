/*
*
* Author: Aristotle Martin
* rocSHMEM code implementating a 2D lid-driven cavity flow.
* D2Q9 lattice
* 
*/

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <sys/time.h>
#include <stddef.h>

#define _STENCILSIZE_ 9
#define _LX_ 256
#define _LY_ 256
#define _NDIMS_ 2
#define _INVALID_ -1
#define _HALO_ 1

#define CHECK_HIP(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }

using namespace rocshmem;
namespace cg = cooperative_groups;

__constant__ double omega_gpu;
__constant__ double uLid_gpu;
__constant__ int icx_gpu[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};
__constant__ int icy_gpu[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1};
__constant__ double w_gpu[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
__constant__ int opp_gpu[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};

__device__ int getGridIdx_gpu(int i, int j, int my_jmin, int dimJ) {
    int relI = i;
    int relJ = j - my_jmin;
    if (relI < 0 || relJ < 0) {
        return _INVALID_;
    }
    if (relI >= _LX_ || relJ >= dimJ) {
        return _INVALID_;
    }
    return i + _LX_ * relJ;
}

__global__ void simLoop(double* distr, double* distrAdv, int* stencilOpPt, double* haloBuff, int msgSize, int nbr_upper, int nbr_lower, int my_jmin_own, int my_jmax_own, int my_jmin_ext, int my_jmax_ext, int numpts, int maxT, int my_pe, int my_ly, int npes) {
    rocshmem_wg_init();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
     // Get the grid we want to synchronize
    auto g = cg::this_grid();

    // initialize the fluid arrays
    for (int ii = tid; ii < numpts; ii+= gridDim.x * blockDim.x) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_ + my_jmin_own;
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            int nbrI = myI + icx_gpu[iNbr];
            int nbrJ = myJ + icy_gpu[iNbr];
            int nbrIJ = getGridIdx_gpu(nbrI, nbrJ, my_jmin_own, my_ly);
            if (nbrIJ < 0) { // check if beyond PE's domain
                stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii * _STENCILSIZE_ + opp_gpu[iNbr]; 
            }
            else {
                // check for walls
                if (nbrI < 0 || nbrI >= _LX_ || nbrJ < 0 || nbrJ >= _LY_) {
                    stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii * _STENCILSIZE_ + opp_gpu[iNbr];
                }
                else {
                    stencilOpPt[ii*_STENCILSIZE_+iNbr] = nbrIJ * _STENCILSIZE_ + iNbr;
                }
            }
            distr[ii*_STENCILSIZE_+iNbr] = w_gpu[iNbr];
            distrAdv[ii*_STENCILSIZE_+iNbr] = 0.0;
        }
    }

    // Synchronize the entire grid
    g.sync();

    // begin time stepping
    for (int t=0; t<maxT; t++) {
        // COLLIDE STREAM
        for (int ii = tid; ii < numpts; ii+= gridDim.x * blockDim.x) {
            int myI = ii % _LX_;
            int myJ = ii / _LX_ + my_jmin_own;
            double rho = 0.0;
            double ux = 0.0;
            double uy = 0.0;
            double distr_local[_STENCILSIZE_];

            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                distr_local[iNbr] = distr[ii*_STENCILSIZE_+iNbr];
                rho += distr_local[iNbr];
                ux += distr_local[iNbr] * icx_gpu[iNbr];
                uy += distr_local[iNbr] * icy_gpu[iNbr];
            }

            double orho = 1.0 / rho;
            ux *= orho;
            uy *= orho;
            double uke = ux * ux + uy * uy;

            // 4. collision + streaming
            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                int nbrInd = stencilOpPt[ii*_STENCILSIZE_+iNbr];
                double cdotu = icx_gpu[iNbr]*ux + icy_gpu[iNbr]*uy;
                double distr_eq = w_gpu[iNbr] * rho * (1 + 3*cdotu + 4.5*cdotu*cdotu - 1.5*uke);
                double distr_stream = omega_gpu*distr_eq + (1.0-omega_gpu)*distr_local[iNbr];
                // check if streaming into nbr pe
                if (myJ == my_jmin_own || myJ == my_jmax_own) {
                    int nbrI = myI + icx_gpu[iNbr];
                    int nbrJ = myJ + icy_gpu[iNbr];
                    if (my_pe < npes - 1 && nbrJ == my_jmax_ext) { // put to nbr upper if not at top
                        rocshmem_double_put(&haloBuff[msgSize+_STENCILSIZE_*nbrI+iNbr], &distr_stream, 1, nbr_upper);
                    } else if (my_pe > 0 && nbrJ == my_jmin_ext) { // put to lower nbr if not at bottom
                        rocshmem_double_put(&haloBuff[_STENCILSIZE_*nbrI+iNbr], &distr_stream, 1, nbr_lower);
                    }
                }
                distrAdv[nbrInd] = distr_stream;
            }
        }

        // Synchronize the entire grid
        rocshmem_quiet();
        g.sync();
        // ZOU-HE BCs
        for (int ii = tid; ii < numpts; ii+= gridDim.x * blockDim.x) {
            int myI = ii % _LX_;
            int myJ = ii / _LX_ + my_jmin_own;
            // copy halos
            if (myJ == my_jmin_own || myJ == my_jmax_own) {
                for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                    // check if in nbr PE
                    int oppIdx = opp_gpu[iNbr];
                    int oppNbrI = myI + icx_gpu[oppIdx];
                    int oppNbrJ = myJ + icy_gpu[oppIdx];
                    if (my_pe > 0 && oppNbrJ == my_jmin_ext && oppNbrI >= 0 && oppNbrI < _LX_) { // copy halo from lower nbr -- make sure we're in domain
                        distrAdv[ii*_STENCILSIZE_+iNbr] = haloBuff[msgSize + _STENCILSIZE_*myI+iNbr];
                    } else if (my_pe < npes - 1 && oppNbrJ == my_jmax_ext && oppNbrI >= 0 && oppNbrI < _LX_) { // copy halo from upper nbr -- make sure we're in domain
                        distrAdv[ii*_STENCILSIZE_+iNbr] = haloBuff[_STENCILSIZE_*myI+iNbr];
                    }
                }
            }
            
            if (myJ == 0) { // BC at lid
                double ux = uLid_gpu;
                double uy = 0.0;
                double rho = (1.0/(1.0-uy))*(distrAdv[ii*_STENCILSIZE_+0]+distrAdv[ii*_STENCILSIZE_+1]+distrAdv[ii*_STENCILSIZE_+3]+2*(distrAdv[ii*_STENCILSIZE_+4]+distrAdv[ii*_STENCILSIZE_+7]+distrAdv[ii*_STENCILSIZE_+8]));

                distrAdv[ii*_STENCILSIZE_+2] = distrAdv[ii*_STENCILSIZE_+4] + (2.0/3.0)*rho*uy;
                distrAdv[ii*_STENCILSIZE_+5] = distrAdv[ii*_STENCILSIZE_+7] - (1.0/2.0)*(distrAdv[ii*_STENCILSIZE_+1] - distrAdv[ii*_STENCILSIZE_+3]) + (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
                distrAdv[ii*_STENCILSIZE_+6] = distrAdv[ii*_STENCILSIZE_+8] + (1.0/2.0)*(distrAdv[ii*_STENCILSIZE_+1] - distrAdv[ii*_STENCILSIZE_+3]) - (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
            }
        }
        g.sync();
        // swap
        for (int ii = tid; ii < numpts; ii+= gridDim.x * blockDim.x) {
            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                double temp = distr[ii*_STENCILSIZE_+iNbr];
                distr[ii*_STENCILSIZE_+iNbr] = distrAdv[ii*_STENCILSIZE_+iNbr];
                distrAdv[ii*_STENCILSIZE_+iNbr] = temp;
            }
        }
        g.sync();
        // begin next time step
    }
    __syncthreads();
    rocshmem_wg_finalize();
}

double shmem_wtime(void) {
    double wtime = 0.0;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    wtime = tv.tv_sec;
    wtime += (double)tv.tv_usec / 1.0e6;
    return wtime;
}

int getGridIdx(int i, int j, int my_jmin, int dimJ) {
    int relI = i;
    int relJ = j - my_jmin;
    if (relI < 0 || relJ < 0) {
        return _INVALID_;
    }
    if (relI >= _LX_ || relJ >= dimJ) {
        return _INVALID_;
    }
    return i + _LX_ * relJ;
}

void writeOutput(double* distr, int* icx, int* icy, int my_pe, int my_jmin_own, int my_jmax_own, int my_ly) {
    std::string file_name = "out_" + std::to_string(my_pe) + ".txt";
    std::ofstream out_file(file_name);

    for (int idxJ=my_jmin_own; idxJ<=my_jmax_own; idxJ++) {
        for (int idxI=0; idxI<_LX_; idxI++) {
            int idxIJ = getGridIdx(idxI,idxJ,my_jmin_own,my_ly);
            // calculate macroscopic quantities
            double rho = 0.0;
            double ux = 0.0;
            double uy = 0.0;
            double distr_local[_STENCILSIZE_];
            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                distr_local[iNbr] = distr[idxIJ*_STENCILSIZE_+iNbr];
                rho += distr_local[iNbr];
                ux += distr_local[iNbr] * icx[iNbr];
                uy += distr_local[iNbr] * icy[iNbr];
            }
            double orho = 1.0 / rho;
            ux *= orho;
            uy *= orho;
            out_file << std::setprecision(16) << idxI << ", " << idxJ << ", " << ux << ", " << uy << ", " << rho << std::endl;
        }
    }

    out_file.close();
}

int main() {
    int my_pe, npes; //declare variables for both pe id of processor and the number of pes
    my_pe = rocshmem_my_pe(); //obtain the pe id number
    int ndevices, my_device = 0;
    CHECK_HIP(hipGetDeviceCount(&ndevices));
    my_device = my_pe % ndevices;
    CHECK_HIP(hipSetDevice(my_device));
    rocshmem_init();
    npes = rocshmem_n_pes(); //obtain the number of pes that can be used

    // set domain bounds and halo cells
    int my_jmin_own, my_jmax_own;
    int my_jmin_ext, my_jmax_ext;
    my_jmin_own = round(_LY_ / npes) * my_pe;
    my_jmax_own = round(_LY_ / npes) * (my_pe + 1) - 1;
    if (my_jmin_own < 0) my_jmin_own = 0;
    if (my_jmax_own > _LY_ - 1) my_jmax_own = _LY_ - 1;
    if (my_pe == npes - 1) my_jmax_own = _LY_ - 1;
    my_jmin_ext = my_jmin_own - _HALO_ >= 0 ? my_jmin_own - _HALO_ : 0;
    my_jmax_ext = my_jmax_own + _HALO_ < _LY_ ? my_jmax_own + _HALO_ : _LY_ - 1;

    int my_ly = my_jmax_own - my_jmin_own + 1;
    int my_ly_ext = my_jmax_ext - my_jmin_ext + 1;
    int msgSize = _LX_ * _STENCILSIZE_;
    int nbr_upper = my_pe + 1 < npes ? my_pe + 1 : 0;
    int nbr_lower = my_pe - 1 >= 0 ? my_pe - 1 : npes - 1;

    std::cout << "PE: " << my_pe << ", nbr_upper = " << nbr_upper << ", nbr_lower = " << nbr_lower << std::endl;
    std::cout << "PE: " << my_pe << ", my_jmin_own = " << my_jmin_own << ", my_jmax_own = " << my_jmax_own << std::endl;
    std::cout << "PE: " << my_pe << ", my_jmin_ext = " << my_jmin_ext << ", my_jmax_ext = " << my_jmax_ext << std::endl;

    // allocate the halo buffer in symmetric memory
    double* haloBuff = (double*)rocshmem_malloc(2*msgSize*sizeof(double));

    int maxT = 10000; // total number of iterations

    double uLid = 0.05; // horizontal lid velocity
    double Re = 100.0; // Reynolds number

    double nu = uLid * _LX_ / Re; // kinematic viscosity
    double omega = 1.0 / (3.0*nu+0.5); // relaxation parameter

    /* GPU buffers */
    double* distr;
    double* distrAdv;
    int icx[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};
    int icy[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1}; 
    double w[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
    int opp[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};

    int* stencilOpPt;
    
    hipMallocManaged((void**)&distr, _LX_ * my_ly * _STENCILSIZE_ * sizeof(double));
    hipMallocManaged((void**)&distrAdv, _LX_ * my_ly * _STENCILSIZE_ * sizeof(double));
    hipMallocManaged((void**)&stencilOpPt, _LX_ * my_ly * _STENCILSIZE_ * sizeof(int));
    hipMemcpyToSymbol(omega_gpu, &omega, sizeof(double));
    hipMemcpyToSymbol(uLid_gpu, &uLid, sizeof(double));

    int numpts = my_ly * _LX_;

    // compute max occupancy for cooperative kernel launch
    int THREADS;
    int BLOCKS;
    hipOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, simLoop, 0, 0);
    std::cout << "max blocks: " << BLOCKS << std::endl;
    std::cout << "max threads: " << THREADS << std::endl;

    double tStart = shmem_wtime();
    // launch cooperative kernel
    void* kernelArgs[] = {(void*)&distr, (void*)&distrAdv, (void*)&stencilOpPt, (void*)&haloBuff, (void*)&msgSize, (void*)&nbr_upper, (void*)&nbr_lower, (void*)&my_jmin_own, (void*)&my_jmax_own, (void*)&my_jmin_ext, (void*)&my_jmax_ext, (void*)&numpts, (void*)&maxT, (void*)&my_pe, (void*)&my_ly, (void*)&npes};
    hipLaunchCooperativeKernel(
        (void*)simLoop,
        BLOCKS,THREADS,
        kernelArgs,0,hipStreamDefault
    );

    rocshmem_barrier_all();
    CHECK_HIP(hipDeviceSynchronize());
    double tEnd = shmem_wtime();
    double tInterval = tEnd - tStart;
    // report max,min,avg runtime among PEs
    double* timers = (double*)rocshmem_malloc(3*sizeof(double));
    timers[0] = tInterval;
    timers[1] = tInterval;
    timers[2] = tInterval;
    rocshmem_ctx_double_max_reduce(ROCSHMEM_CTX_DEFAULT,ROCSHMEM_TEAM_WORLD, &timers[0], &timers[0], 1);
    rocshmem_ctx_double_min_reduce(ROCSHMEM_CTX_DEFAULT,ROCSHMEM_TEAM_WORLD, &timers[1], &timers[1], 1);
    rocshmem_ctx_double_sum_reduce(ROCSHMEM_CTX_DEFAULT,ROCSHMEM_TEAM_WORLD, &timers[2], &timers[2], 1);
    timers[2] = timers[2] / npes;
    if (my_pe == 0)
    {
        std::cout << "Max runtime: " << timers[0] << std::endl;
        std::cout << "Min runtime: " << timers[1] << std::endl;
        std::cout << "Average runtime: " << timers[2] << std::endl;
    }
    // write output
    writeOutput(distr,icx,icy,my_pe,my_jmin_own,my_jmax_own,my_ly);
    rocshmem_finalize();
}   