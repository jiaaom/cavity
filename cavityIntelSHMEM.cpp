/*
*
* Author: Aristotle Martin
* Intel SHMEM (ISHMEM) code implementing a 2D lid-driven cavity flow.
* D2Q9 lattice
* 
*/

#include <sycl/sycl.hpp>
#include <ishmem.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <sys/time.h>
#include <stddef.h>

#define _STENCILSIZE_ 9
#define _LX_ 2048
#define _LY_ 2048
#define _NDIMS_ 2
#define _INVALID_ -1
#define _HALO_ 1

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

void initializeFluidKernel(double* distr, double* distrAdv, int* stencilOpPt,
                            int my_jmin_own, int my_ly, int numpts,
                            int* icx, int* icy, int* opp, double* w,
                            sycl::nd_item<1> item) {
    // initialize the fluid arrays
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);

    for (int ii = tid; ii < numpts; ii+= item.get_group_range(0) * item.get_local_range(0)) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_ + my_jmin_own;
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            int nbrI = myI + icx[iNbr];
            int nbrJ = myJ + icy[iNbr];
            int nbrIJ = getGridIdx(nbrI, nbrJ, my_jmin_own, my_ly);
            if (nbrIJ < 0) { // check if beyond PE's domain
                stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii * _STENCILSIZE_ + opp[iNbr]; 
            }
            else {
                // check for walls
                if (nbrI < 0 || nbrI >= _LX_ || nbrJ < 0 || nbrJ >= _LY_) {
                    stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii * _STENCILSIZE_ + opp[iNbr];
                }
                else {
                    stencilOpPt[ii*_STENCILSIZE_+iNbr] = nbrIJ * _STENCILSIZE_ + iNbr;
                }
            }
            distr[ii*_STENCILSIZE_+iNbr] = w[iNbr];
            distrAdv[ii*_STENCILSIZE_+iNbr] = 0.0;
        }
    }
}

void collideStreamKernel(double* distr, double* distrAdv, int* stencilOpPt, double* haloBuff, int msgSize,
                            int nbr_upper, int nbr_lower, int my_jmin_own, int my_jmax_own, int my_jmin_ext,
                            int my_jmax_ext, int numpts, int my_pe, int my_ly, int npes,
                            int* icx, int* icy, double* w, double omega,
                            sycl::nd_item<1> item)
{
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);

    for (int ii = tid; ii < numpts; ii+= item.get_group_range(0) * item.get_local_range(0)) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_ + my_jmin_own;
        double rho = 0.0;
        double ux = 0.0;
        double uy = 0.0;
        double distr_local[_STENCILSIZE_];

        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            distr_local[iNbr] = distr[ii*_STENCILSIZE_+iNbr];
            rho += distr_local[iNbr];
            ux += distr_local[iNbr] * icx[iNbr];
            uy += distr_local[iNbr] * icy[iNbr];
        }

        double orho = 1.0 / rho;
        ux *= orho;
        uy *= orho;
        double uke = ux * ux + uy * uy;

        // 4. collision + streaming
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            int nbrInd = stencilOpPt[ii*_STENCILSIZE_+iNbr];
            double cdotu = icx[iNbr]*ux + icy[iNbr]*uy;
            double distr_eq = w[iNbr] * rho * (1 + 3*cdotu + 4.5*cdotu*cdotu - 1.5*uke);
            distrAdv[nbrInd] = omega*distr_eq + (1.0-omega)*distr_local[iNbr];
            // check if streaming into nbr pe
            if (myJ == my_jmin_own || myJ == my_jmax_own) {
                int nbrI = myI + icx[iNbr];
                int nbrJ = myJ + icy[iNbr];
                if (my_pe < npes - 1 && nbrJ == my_jmax_ext) { // put to nbr upper if not at top
                    ishmem_double_put(&haloBuff[msgSize+_STENCILSIZE_*nbrI+iNbr], &distrAdv[nbrInd], 1, nbr_upper);
                } else if (my_pe > 0 && nbrJ == my_jmin_ext) { // put to lower nbr if not at bottom
                    ishmem_double_put(&haloBuff[_STENCILSIZE_*nbrI+iNbr], &distrAdv[nbrInd], 1, nbr_lower);
                }
            }
        }
    }
}

void zouHeBCKernel(double* distr, double* distrAdv, int* stencilOpPt, double* haloBuff,
                    int msgSize, int nbr_upper, int nbr_lower, int my_jmin_own,
                    int my_jmax_own, int my_jmin_ext, int my_jmax_ext, int numpts,
                    int my_pe, int my_ly, int npes,
                    int* icx,
                    int* icy,
                    int* opp,
                    double uLid,
                    sycl::nd_item<1> item) {
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);

    for (int ii = tid; ii < numpts; ii+= item.get_group_range(0) * item.get_local_range(0)) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_ + my_jmin_own;
        // copy halos
        if (myJ == my_jmin_own || myJ == my_jmax_own) {
            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                // check if in nbr PE
                int oppIdx = opp[iNbr];
                int oppNbrI = myI + icx[oppIdx];
                int oppNbrJ = myJ + icy[oppIdx];
                if (my_pe > 0 && oppNbrJ == my_jmin_ext && oppNbrI >= 0 && oppNbrI < _LX_) { // copy halo from lower nbr -- make sure we're in domain
                    distrAdv[ii*_STENCILSIZE_+iNbr] = haloBuff[msgSize + _STENCILSIZE_*myI+iNbr];
                } else if (my_pe < npes - 1 && oppNbrJ == my_jmax_ext && oppNbrI >= 0 && oppNbrI < _LX_) { // copy halo from upper nbr -- make sure we're in domain
                    distrAdv[ii*_STENCILSIZE_+iNbr] = haloBuff[_STENCILSIZE_*myI+iNbr];
                }
            }
        }
        
        if (myJ == 0) { // BC at lid
            double ux = uLid;
            double uy = 0.0;
            double rho = (1.0/(1.0-uy))*(distrAdv[ii*_STENCILSIZE_+0]+distrAdv[ii*_STENCILSIZE_+1]+distrAdv[ii*_STENCILSIZE_+3]+2*(distrAdv[ii*_STENCILSIZE_+4]+distrAdv[ii*_STENCILSIZE_+7]+distrAdv[ii*_STENCILSIZE_+8]));

            distrAdv[ii*_STENCILSIZE_+2] = distrAdv[ii*_STENCILSIZE_+4] + (2.0/3.0)*rho*uy;
            distrAdv[ii*_STENCILSIZE_+5] = distrAdv[ii*_STENCILSIZE_+7] - (1.0/2.0)*(distrAdv[ii*_STENCILSIZE_+1] - distrAdv[ii*_STENCILSIZE_+3]) + (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
            distrAdv[ii*_STENCILSIZE_+6] = distrAdv[ii*_STENCILSIZE_+8] + (1.0/2.0)*(distrAdv[ii*_STENCILSIZE_+1] - distrAdv[ii*_STENCILSIZE_+3]) - (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
        }
    }
}

double shmem_wtime(void) {
    double wtime = 0.0;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    wtime = tv.tv_sec;
    wtime += (double)tv.tv_usec / 1.0e6;
    return wtime;
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
    ishmem_init(); // Initialize ISHMEM
    int my_pe = ishmem_my_pe(); //obtain the pe id number
    int npes = ishmem_n_pes(); //obtain the number of pes that can be used

    sycl::queue q;

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
    double* haloBuff = (double*)ishmem_malloc(2*msgSize*sizeof(double));

    int maxT = 10000; // total number of iterations

    double uLid = 0.05; // horizontal lid velocity
    double Re = 100.0; // Reynolds number

    double nu = uLid * _LX_ / Re; // kinematic viscosity
    double omega = 1.0 / (3.0*nu+0.5); // relaxation parameter

    /* setup GPU buffers */
    double* distr;
    double* distrAdv;
    int* stencilOpPt;
    int icx[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};
    int icy[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1}; 
    int opp[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};
    double w[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
    
    int* icx_gpu = sycl::malloc_shared<int>(_STENCILSIZE_, q);
    int* icy_gpu = sycl::malloc_shared<int>(_STENCILSIZE_, q);
    int* opp_gpu = sycl::malloc_shared<int>(_STENCILSIZE_, q);
    double* w_gpu = sycl::malloc_shared<double>(_STENCILSIZE_, q);
    q.memcpy(icx_gpu,&icx[0],_STENCILSIZE_*sizeof(int)).wait();
    q.memcpy(icy_gpu,&icy[0],_STENCILSIZE_*sizeof(int)).wait();
    q.memcpy(opp_gpu,&opp[0],_STENCILSIZE_*sizeof(int)).wait();
    q.memcpy(w_gpu,&w[0],_STENCILSIZE_*sizeof(double)).wait();

    distr = sycl::malloc_shared<double>(_LX_ * my_ly * _STENCILSIZE_, q);
    // allocate distrAdv in symmetric memory
    distrAdv = (double*)ishmem_malloc(_LX_ * my_ly * _STENCILSIZE_ *sizeof(double));
    stencilOpPt = sycl::malloc_shared<int>(_LX_ * my_ly * _STENCILSIZE_, q);

    int numpts = my_ly * _LX_;

    // compute kernel dimensions
    const sycl::range<3> dimBlock = sycl::range<3>{1,1,128};
    const int gridDimX = ceil(numpts / (double)dimBlock[2]);
    const int gridDimY = 1;
    sycl::range<3> dimGrid(1, gridDimY, gridDimX);

    double tStart = shmem_wtime();
    q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                initializeFluidKernel(distr,distrAdv,stencilOpPt,
                                        my_jmin_own,my_ly,numpts,
                                        icx_gpu,icy_gpu,opp_gpu,
                                        w_gpu,item);
    }).wait();
    for (int t=0; t<maxT; t++) {
        q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                collideStreamKernel(distr,distrAdv,stencilOpPt,
                                        haloBuff,msgSize,nbr_upper,
                                        nbr_lower,my_jmin_own,my_jmax_own,
                                        my_jmin_ext,my_jmax_ext,
                                        numpts,my_pe,my_ly,
                                        npes,icx_gpu,icy_gpu,
                                        w_gpu,omega,item);
        }).wait();
        q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
            zouHeBCKernel(distr,distrAdv,stencilOpPt,
                                    haloBuff,msgSize,nbr_upper,
                                    nbr_lower,my_jmin_own,my_jmax_own,
                                    my_jmin_ext,my_jmax_ext,
                                    numpts,my_pe,my_ly,
                                    npes,icx_gpu,icy_gpu,
                                    opp_gpu,uLid,
                                    item);
        }).wait();
        std::swap(distr,distrAdv);
    }
    ishmem_barrier_all();
    double tEnd = shmem_wtime();

    double tInterval = tEnd - tStart;
    // report max,min,avg runtime among PEs
    double timers_h[3] = {tInterval,tInterval,tInterval};
    double* timers = (double*)ishmem_malloc(3*sizeof(double));
    q.memcpy(timers,&timers_h[0],3*sizeof(double)).wait();
    ishmem_double_max_reduce(ISHMEM_TEAM_WORLD, &timers[0], &timers[0], 1);
    ishmem_double_min_reduce(ISHMEM_TEAM_WORLD, &timers[1], &timers[1], 1);
    ishmem_double_sum_reduce(ISHMEM_TEAM_WORLD, &timers[2], &timers[2], 1);
    q.memcpy(&timers_h[0],timers,3*sizeof(double)).wait();
    timers_h[2] = timers_h[2] / npes;
    if (my_pe == 0)
    {
        std::cout << "Max runtime: " << timers_h[0] << std::endl;
        std::cout << "Min runtime: " << timers_h[1] << std::endl;
        std::cout << "Average runtime: " << timers_h[2] << std::endl;
    }
    // write output
    writeOutput(distr,icx,icy,my_pe,my_jmin_own,my_jmax_own,my_ly);
    ishmem_finalize();
}   