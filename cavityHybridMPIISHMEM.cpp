/*
*
* Author: Aristotle Martin
* SYCL implementation of 2D cavity flow with MPI + SYCL (shared memory)
* 
* D2Q9 lattice
* 
*/
#include <sycl/sycl.hpp>
#include <mpi.h>
#include <ishmemx.h>
#include <ishmem.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <sys/time.h>
#include <stddef.h>
#include <map>

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

void setupIntraNodeNbrMap(int rank, int size, int nbr_up, int nbr_down, std::map<int,int>& intraNodeNbrs) {
    intraNodeNbrs.clear();
    int* nodeLengths = (int*)calloc(size,sizeof(int));
    char** nodeNames = (char**)malloc(size*sizeof(char*));
    for (int i=0; i<size; i++) {
        nodeNames[i] = (char*)calloc((MPI_MAX_PROCESSOR_NAME),sizeof(char));
    }
    MPI_Get_processor_name(nodeNames[rank], &nodeLengths[rank]);
    // comm node names
    // first, exchange sizes
    MPI_Allreduce(MPI_IN_PLACE, nodeLengths, size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // exchange data
    int tag=41;
    int iNbr;
    MPI_Request* req = new MPI_Request[2*(size-1)];
    int count = 0;

    for (iNbr=0; iNbr<size; iNbr++) {
        if (iNbr != rank) {
            MPI_Irecv(nodeNames[iNbr], nodeLengths[iNbr], MPI_CHAR, iNbr, tag, MPI_COMM_WORLD, req + count);
            count++;
        }
    }

    for (iNbr=0; iNbr<size; iNbr++) {
        if (iNbr != rank) {
            MPI_Isend(nodeNames[rank], nodeLengths[rank], MPI_CHAR, iNbr, tag, MPI_COMM_WORLD, req + count);
            count++;
        }
    }
    MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
    
    if (strcmp(nodeNames[rank], nodeNames[nbr_up]) == 0) {
        intraNodeNbrs[nbr_up] = nbr_up;
    }
    if (strcmp(nodeNames[rank], nodeNames[nbr_down]) == 0) {
        intraNodeNbrs[nbr_down] = nbr_down;
    }
}

void copyToHaloBufferKernel(double* distr, double* bufferSend, int* locsToSendGPU, int commLength, sycl::nd_item<1> item) {
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);
    for (int ii = tid; ii < commLength; ii+=item.get_group_range(0) * item.get_local_range(0)) {
        int sendLoc = locsToSendGPU[ii];
        bufferSend[ii] = distr[sendLoc];
    }
}

void copyFromHaloBufferKernel(double* distr, double* bufferRecv, int* locsToRecvGPU, int commLength, sycl::nd_item<1> item) {
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);
    for (int ii = tid; ii < commLength; ii+=item.get_group_range(0) * item.get_local_range(0)) {
        int recvLoc = locsToRecvGPU[ii];
        distr[recvLoc] = bufferRecv[ii];
    }
}

void commHalos(double* distr, int* locsToSendGPU, int* locsToRecvGPU, double* bufferSend, double* bufferRecv, int nLocsSendToLower, int nLocsSendToUpper, int nLocsRecvFromLower, int nLocsRecvFromUpper, int myRank, sycl::queue& q) {
    
    int commLength = nLocsSendToUpper + nLocsSendToLower;
    // compute kernel dimensions
    const sycl::range<3> dimBlock = sycl::range<3>{1,1,128};
    const int gridDimX = ceil(commLength / (double)dimBlock[2]);
    const int gridDimY = 1;
    sycl::range<3> dimGrid(1, gridDimY, gridDimX);

    q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                copyToHaloBufferKernel(distr,bufferSend,locsToSendGPU,
                                        commLength,item);
    }).wait();

    MPI_Request* sendRequest = new MPI_Request[2];
    MPI_Request* recvRequest = new MPI_Request[2];
    int tag = 44;
    int sendCount = 0;
    int recvCount = 0;
    // lower nbr
    if (nLocsRecvFromLower > 0) {
        MPI_Irecv(&bufferRecv[0], nLocsRecvFromLower, MPI_DOUBLE, myRank-1, tag, MPI_COMM_WORLD, recvRequest+recvCount);
        recvCount++;
        MPI_Isend(&bufferSend[0], nLocsSendToLower, MPI_DOUBLE, myRank-1, tag, MPI_COMM_WORLD, sendRequest+sendCount);
        sendCount++;
    }
    // upper nbr
    if (nLocsRecvFromUpper > 0) {
        int recvOffset = nLocsRecvFromLower;
        int sendOffset = nLocsSendToLower;
        MPI_Irecv(&bufferRecv[recvOffset],nLocsRecvFromUpper,MPI_DOUBLE,myRank+1,tag,MPI_COMM_WORLD,recvRequest+recvCount);
        recvCount++;
        MPI_Isend(&bufferSend[sendOffset],nLocsSendToUpper,MPI_DOUBLE,myRank+1,tag,MPI_COMM_WORLD,sendRequest+sendCount);
        sendCount++;
    }
    MPI_Waitall(sendCount, sendRequest, MPI_STATUSES_IGNORE);
    MPI_Waitall(recvCount, recvRequest, MPI_STATUSES_IGNORE);
    
    q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                copyFromHaloBufferKernel(distr,bufferRecv,locsToRecvGPU,
                                        commLength,item);
    }).wait();
}

void exchangeLocAoS(std::vector<int>& locsToSendAoS, std::vector<int>& locsToRecvAoS, std::vector<int>& locSendCounts, std::vector<int>& locRecvCounts, int myRank) {
    int nLocsRecvFromLower = locRecvCounts[0];
    int nLocsRecvFromUpper = locRecvCounts[1];
    int nTotRecvLocs = nLocsRecvFromLower + nLocsRecvFromUpper;
    locsToRecvAoS.resize(nTotRecvLocs*3);
    int nLocsSendToLower = locSendCounts[0];
    int nLocsSendToUpper = locSendCounts[1];
    // sanity checks
    int nTotSendLocs = nLocsSendToLower + nLocsSendToUpper;
    assert(locsToSendAoS.size() == nTotSendLocs*3);
    assert(locsToSendAoS.size() == locsToRecvAoS.size());

    MPI_Request* sendRequest = new MPI_Request[2];
    MPI_Request* recvRequest = new MPI_Request[2];
    int tag = 43;
    int sendCount = 0;
    int recvCount = 0;
    // lower nbr
    if (nLocsRecvFromLower > 0) {
        MPI_Irecv(&locsToRecvAoS[0], nLocsRecvFromLower*3, MPI_INT, myRank-1, tag, MPI_COMM_WORLD, recvRequest+recvCount);
        recvCount++;
        MPI_Isend(&locsToSendAoS[0], nLocsSendToLower*3, MPI_INT, myRank-1, tag, MPI_COMM_WORLD, sendRequest+sendCount);
        sendCount++;
    }
    // upper nbr
    int recvOffset = nLocsRecvFromLower*3;
    int sendOffset = nLocsSendToLower*3;
    if (nLocsRecvFromUpper > 0) {
        MPI_Irecv(&locsToRecvAoS[recvOffset],nLocsRecvFromUpper*3,MPI_INT,myRank+1,tag,MPI_COMM_WORLD,recvRequest+recvCount);
        recvCount++;
        MPI_Isend(&locsToSendAoS[sendOffset],nLocsSendToUpper*3,MPI_INT,myRank+1,tag,MPI_COMM_WORLD,sendRequest+sendCount);
        sendCount++;
    }
    MPI_Waitall(sendCount, sendRequest, MPI_STATUSES_IGNORE);
    MPI_Waitall(recvCount, recvRequest, MPI_STATUSES_IGNORE);
}

void setupSendLocs(std::vector<int>& locsToSend, std::vector<int>& locsToSendAoS, std::vector<int>& locSendCounts, int myRank, int nRanks, int my_jmin_own, int my_jmax_own, int my_jmin_ext, int my_jmax_ext, int my_ly_ext, int* icx, int* icy) {
    // locsToSend: the ii's in distr that we are sending to each nbr
    int commLength = 0;
    int nLocsSendLower = 0;
    int nLocsSendUpper = 0;
    // lower nbr first
    if (myRank > 0) {
        int myJ = my_jmin_own;
        for (int myI=0; myI < _LX_; myI++) {
            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                int nbrI = myI + icx[iNbr];
                int nbrJ = myJ + icy[iNbr];
                if ((nbrI >= 0 && nbrI < _LX_) && nbrJ == my_jmin_ext) {
                    locsToSendAoS.push_back(nbrI);
                    locsToSendAoS.push_back(nbrJ);
                    locsToSendAoS.push_back(iNbr);
                    int nbrIJ = getGridIdx(nbrI,nbrJ,my_jmin_ext,my_ly_ext);
                    int nbrInd = nbrIJ * _STENCILSIZE_ + iNbr;
                    assert(nbrInd > 0);
                    locsToSend.push_back(nbrInd);
                    commLength++;
                    locSendCounts[0]++;
                }
            }
        }
    }

    // now upper nbr
    if (myRank < nRanks - 1) {
        int myJ = my_jmax_own;
        for (int myI=0; myI < _LX_; myI++) {
            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                int nbrI = myI + icx[iNbr];
                int nbrJ = myJ + icy[iNbr];
                if ((nbrI >= 0 && nbrI < _LX_) && nbrJ == my_jmax_ext) {
                    locsToSendAoS.push_back(nbrI);
                    locsToSendAoS.push_back(nbrJ);
                    locsToSendAoS.push_back(iNbr);
                    int nbrIJ = getGridIdx(nbrI,nbrJ,my_jmin_ext,my_ly_ext);
                    int nbrInd = nbrIJ * _STENCILSIZE_ + iNbr;
                    assert(nbrInd > 0);
                    locsToSend.push_back(nbrInd);
                    commLength++;
                    locSendCounts[1]++;
                }
            }
        }
    } 
}

void setupRecvLocs(std::vector<int>& locsToRecv, std::vector<int>& locsToRecvAoS, std::vector<int>& locRecvCounts, int my_jmin_ext, int my_jmin_own, int my_ly_ext) {
    int nLocsRecvFromLower = locRecvCounts[0];
    int nLocsRecvFromUpper = locRecvCounts[1];
    int nTotRecvLocs = nLocsRecvFromLower + nLocsRecvFromUpper;
    if (nLocsRecvFromLower > 0) {
        for (int lowerIdx = 0; lowerIdx < nLocsRecvFromLower; lowerIdx++) {
            int recvI = locsToRecvAoS[lowerIdx*3];
            int recvJ = locsToRecvAoS[lowerIdx*3+1];
            int iNbr = locsToRecvAoS[lowerIdx*3+2];
            int ii = getGridIdx(recvI,recvJ,my_jmin_ext,my_ly_ext);
            int recvInd = ii*_STENCILSIZE_+iNbr;
            locsToRecv.push_back(recvInd);
        }
    }
    if (nLocsRecvFromUpper > 0) {
        for (int upperIdx = 0; upperIdx < nLocsRecvFromUpper; upperIdx++) {
            int recvI = locsToRecvAoS[(upperIdx+nLocsRecvFromLower)*3];
            int recvJ = locsToRecvAoS[(upperIdx+nLocsRecvFromLower)*3+1];
            int iNbr = locsToRecvAoS[(upperIdx+nLocsRecvFromLower)*3+2];
            int ii = getGridIdx(recvI,recvJ,my_jmin_ext,my_ly_ext);
            int recvInd = ii*_STENCILSIZE_+iNbr;
            locsToRecv.push_back(recvInd);
        }
    }
    // sanity check
    assert(locsToRecv.size() == nTotRecvLocs);
}

void exchangeLocSizes(std::vector<int>& locSendCounts, std::vector<int>& locRecvCounts, int myRank, int nRanks) {
    
    MPI_Request* sendRequest = new MPI_Request[2];
    MPI_Request* recvRequest = new MPI_Request[2];
    int tag = 42;
    int sendCount = 0;
    int recvCount = 0;
    // lower nbr
    if (myRank > 0) {
        MPI_Irecv(&locRecvCounts[0], 1, MPI_INT, myRank-1, tag, MPI_COMM_WORLD, recvRequest+recvCount);
        recvCount++;
        MPI_Isend(&locSendCounts[0], 1, MPI_INT, myRank-1, tag, MPI_COMM_WORLD, sendRequest+sendCount);
        sendCount++;
    }
    // upper nbr
    if (myRank < nRanks - 1) {
        MPI_Irecv(&locRecvCounts[1], 1, MPI_INT, myRank+1, tag, MPI_COMM_WORLD, recvRequest+recvCount);
        recvCount++;
        MPI_Isend(&locSendCounts[1], 1, MPI_INT, myRank+1, tag, MPI_COMM_WORLD, sendRequest+sendCount);
        sendCount++;
    }
    MPI_Waitall(sendCount, sendRequest, MPI_STATUSES_IGNORE);
    MPI_Waitall(recvCount, recvRequest, MPI_STATUSES_IGNORE);
}

void initializeFluidKernel(double* distr, double* distrAdv, int* stencilOpPt,
                            int my_jmin_ext, int my_ly_ext, int numpts,
                            int* icx, int* icy, int* opp, double* w,
                            sycl::nd_item<1> item) {
    // initialize the fluid arrays
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);

    for (int ii = tid; ii < numpts; ii+= item.get_group_range(0) * item.get_local_range(0)) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_ + my_jmin_ext;
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            int nbrI = myI + icx[iNbr];
            int nbrJ = myJ + icy[iNbr];
            int nbrIJ = getGridIdx(nbrI, nbrJ, my_jmin_ext, my_ly_ext);
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

void collideStreamSHMEMKernel(double* distr, double* distrAdv, int* stencilOpPt, double* haloBuff, int msgSize,
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
                    ishmem_double_p(&haloBuff[msgSize+_STENCILSIZE_*nbrI+iNbr], distrAdv[nbrInd], nbr_upper);
                } else if (my_pe > 0 && nbrJ == my_jmin_ext) { // put to lower nbr if not at bottom
		            ishmem_double_p(&haloBuff[_STENCILSIZE_*nbrI+iNbr], distrAdv[nbrInd], nbr_lower);
                }
            }
        }
    }
}

void collideStreamMPIKernel(double* distr, double* distrAdv, int* stencilOpPt, int msgSize,
                            int nbr_upper, int nbr_lower, int my_jmin_own, int my_jmax_own, int my_jmin_ext,
                            int my_jmax_ext, int numpts, int myRank, int my_ly, int nRanks,
                            int* icx, int* icy, double* w, double omega,
                            sycl::nd_item<1> item)
{
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);

    for (int ii = tid; ii < numpts; ii+= item.get_group_range(0) * item.get_local_range(0)) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_ + my_jmin_ext;
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
        }
    }
}

void zouHeBCSHMEMKernel(double* distr, double* distrAdv, int* stencilOpPt, double* haloBuff,
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

void zouHeBCMPIKernel(double* distr, double* distrAdv, int* stencilOpPt,
                    int msgSize, int nbr_upper, int nbr_lower, int my_jmin_own,
                    int my_jmax_own, int my_jmin_ext, int my_jmax_ext, int numpts,
                    int myRank, int my_ly, int nRanks,
                    int* icx,
                    int* icy,
                    int* opp,
                    double uLid,
                    sycl::nd_item<1> item) {
    int tid = item.get_local_id(0) +
        item.get_group(0) * item.get_local_range(0);

    for (int ii = tid; ii < numpts; ii+= item.get_group_range(0) * item.get_local_range(0)) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_ + my_jmin_ext;
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


void writeOutput(double* distr, int* icx, int* icy, int myRank, int my_jmin_ext, int my_jmin_own, int my_jmax_own, int my_ly, int my_ly_ext, bool commSHMEM) {
    std::string file_name = "out_" + std::to_string(myRank) + ".txt";
    std::ofstream out_file(file_name);

    for (int idxJ=my_jmin_own; idxJ<=my_jmax_own; idxJ++) {
        for (int idxI=0; idxI<_LX_; idxI++) {
            int idxIJ;
            if (commSHMEM) {
                idxIJ = getGridIdx(idxI,idxJ,my_jmin_own,my_ly);
            } else {
                idxIJ = getGridIdx(idxI,idxJ,my_jmin_ext,my_ly_ext);
            }
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

int main(int argc, char** argv) {
    int mpi_thread_level_available;
    int mpi_thread_level_required = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, mpi_thread_level_required,
                  &mpi_thread_level_available);
    assert(mpi_thread_level_available >= mpi_thread_level_required);

    int myRank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    sycl::queue q;
    std::cout << "My Rank: " << myRank
              << " , Selected device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    MPI_Comm mpi_comm;
    ishmemx_attr_t attr;
    mpi_comm      = MPI_COMM_WORLD;
    ishmemx_runtime_type_t runtime = ISHMEMX_RUNTIME_MPI;
    attr.runtime = runtime;
    attr.mpi_comm = &mpi_comm;
    ishmemx_init_attr(&attr); // Initialize ISHMEM

    int my_pe = ishmem_my_pe(); //obtain the pe id number
    int npes = ishmem_n_pes(); //obtain the number of pes that can be used

    // set domain bounds and halo cells
    int my_jmin_own, my_jmax_own;
    int my_jmin_ext, my_jmax_ext;
    my_jmin_own = round(_LY_ / nRanks) * myRank;
    my_jmax_own = round(_LY_ / nRanks) * (myRank + 1) - 1;
    if (my_jmin_own < 0) my_jmin_own = 0;
    if (my_jmax_own > _LY_ - 1) my_jmax_own = _LY_ - 1;
    if (myRank == nRanks - 1) my_jmax_own = _LY_ - 1;
    my_jmin_ext = my_jmin_own - _HALO_ >= 0 ? my_jmin_own - _HALO_ : 0;
    my_jmax_ext = my_jmax_own + _HALO_ < _LY_ ? my_jmax_own + _HALO_ : _LY_ - 1;

    int my_ly = my_jmax_own - my_jmin_own + 1;
    int my_ly_ext = my_jmax_ext - my_jmin_ext + 1;
    int msgSize = _LX_ * _STENCILSIZE_;
    int nbr_upper = myRank + 1 < nRanks ? myRank + 1 : 0;
    int nbr_lower = myRank - 1 >= 0 ? myRank - 1 : nRanks - 1;

    std::cout << "Rank: " << myRank << ", nbr_upper = " << nbr_upper << ", nbr_lower = " << nbr_lower << std::endl;
    std::cout << "Rank: " << myRank << ", my_jmin_own = " << my_jmin_own << ", my_jmax_own = " << my_jmax_own << std::endl;
    std::cout << "Rank: " << myRank << ", my_jmin_ext = " << my_jmin_ext << ", my_jmax_ext = " << my_jmax_ext << std::endl;

    // exchange node names among MPI processors. Figure out which nbrs are on the same node.
    std::map<int,int> intraNodeNbrs;
    setupIntraNodeNbrMap(myRank, nRanks, nbr_upper, nbr_lower, intraNodeNbrs);
    bool commSHMEM = false;
    
    //check:
    /*if (intraNodeNbrs.find(nbr_upper) != intraNodeNbrs.end() && intraNodeNbrs.find(nbr_lower) != intraNodeNbrs.end()) {
        std::cout << "Rank: " << myRank << " has both nbrs on the same node!" << std::endl;
        commSHMEM = true;
    } else {
        std::cout << "Rank: " << myRank << " has at least one nbr on a different node!" << std::endl;
    }*/

    if (commSHMEM) my_ly_ext = my_ly;

    // allocate halo buffer in symmetric memmory
    double* haloBuff = (double*)ishmem_malloc(2*msgSize*sizeof(double));

    int maxT = 10000; // total number of iterations

    double uLid = 0.05; // horizontal lid velocity
    double Re = 100.0; // Reynolds number

    double nu = uLid * _LX_ / Re; // kinematic viscosity
    double omega = 1.0 / (3.0*nu+0.5); // relaxation parameter

    /* setup GPU buffers */
    double* distr;
    double* distr_host;
    double* distrAdv;
    int* stencilOpPt;
    int icx[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};
    int icy[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1}; 
    int opp[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};
    double w[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
    
    int* icx_gpu = sycl::malloc_device<int>(_STENCILSIZE_, q);
    int* icy_gpu = sycl::malloc_device<int>(_STENCILSIZE_, q);
    int* opp_gpu = sycl::malloc_device<int>(_STENCILSIZE_, q);
    double* w_gpu = sycl::malloc_device<double>(_STENCILSIZE_, q);
    q.memcpy(icx_gpu,&icx[0],_STENCILSIZE_*sizeof(int)).wait();
    q.memcpy(icy_gpu,&icy[0],_STENCILSIZE_*sizeof(int)).wait();
    q.memcpy(opp_gpu,&opp[0],_STENCILSIZE_*sizeof(int)).wait();
    q.memcpy(w_gpu,&w[0],_STENCILSIZE_*sizeof(double)).wait();

    if (commSHMEM) {
        distr = (double*)ishmem_malloc(_LX_ * my_ly * _STENCILSIZE_ *sizeof(double));
        distrAdv = (double*)ishmem_malloc(_LX_ * my_ly * _STENCILSIZE_ *sizeof(double));
    } else {
        distr = sycl::malloc_device<double>(_LX_ * my_ly_ext * _STENCILSIZE_, q);
        distrAdv = sycl::malloc_device<double>(_LX_ * my_ly_ext * _STENCILSIZE_, q);
    }
    distr_host = sycl::malloc_host<double>(_LX_ * my_ly_ext * _STENCILSIZE_, q);
    stencilOpPt = sycl::malloc_device<int>(_LX_ * my_ly_ext * _STENCILSIZE_, q);
    int numpts = my_ly_ext * _LX_;

    // MPI communication stuff
    std::vector<int> locsToSend;
    std::vector<int> locsToSendAoS;
    std::vector<int> locsToRecv;
    std::vector<int> locsToRecvAoS;
    std::vector<int> locSendCounts(2,0);
    std::vector<int> locRecvCounts(2,0);
    int* locsToSendGPU;
    int* locsToRecvGPU;
    double* bufferSend;
    double* bufferRecv;
    int nLocsRecvFromLower;
    int nLocsRecvFromUpper;
    int nTotRecvLocs;
    int nLocsSendToLower;
    int nLocsSendToUpper;
    int nTotSendLocs;
    if (!commSHMEM) {
        setupSendLocs(locsToSend,locsToSendAoS,locSendCounts,myRank,nRanks,
                        my_jmin_own,my_jmax_own,my_jmin_ext,my_jmax_ext,my_ly_ext,
                        &icx[0],&icy[0]);
        exchangeLocSizes(locSendCounts,locRecvCounts,myRank,nRanks);
        exchangeLocAoS(locsToSendAoS,locsToRecvAoS,locSendCounts,locRecvCounts,myRank);
        setupRecvLocs(locsToRecv,locsToRecvAoS,locRecvCounts,my_jmin_ext,my_jmin_own,my_ly_ext);
        // message buffers
        nLocsRecvFromLower = locRecvCounts[0];
        nLocsRecvFromUpper = locRecvCounts[1];
        nTotRecvLocs = nLocsRecvFromLower + nLocsRecvFromUpper;
        nLocsSendToLower = locSendCounts[0];
        nLocsSendToUpper = locSendCounts[1];
        nTotSendLocs = nLocsSendToLower + nLocsSendToUpper;
        assert(nTotSendLocs == nTotRecvLocs);
        locsToSendGPU = sycl::malloc_device<int>(nTotSendLocs, q);
        locsToRecvGPU = sycl::malloc_device<int>(nTotRecvLocs, q);
        bufferSend = sycl::malloc_device<double>(nTotSendLocs, q);
        bufferRecv = sycl::malloc_device<double>(nTotRecvLocs, q);
        q.memcpy(locsToSendGPU,locsToSend.data(),nTotSendLocs*sizeof(int)).wait();
        q.memcpy(locsToRecvGPU,locsToRecv.data(),nTotRecvLocs*sizeof(int)).wait();
    }

    // compute kernel dimensions
    const sycl::range<3> dimBlock = sycl::range<3>{1,1,128};
    const int gridDimX = ceil(numpts / (double)dimBlock[2]);
    const int gridDimY = 1;
    sycl::range<3> dimGrid(1, gridDimY, gridDimX);

    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = shmem_wtime();;
    q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                initializeFluidKernel(distr,distrAdv,stencilOpPt,
                                        my_jmin_ext,my_ly_ext,numpts,
                                        icx_gpu,icy_gpu,opp_gpu,
                                        w_gpu,item);
    }).wait();
    for (int t=0; t<maxT; t++) {
        if (commSHMEM) {
            q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                    collideStreamSHMEMKernel(distr,distrAdv,stencilOpPt,
                                            haloBuff,msgSize,nbr_upper,
                                            nbr_lower,my_jmin_own,my_jmax_own,
                                            my_jmin_ext,my_jmax_ext,
                                            numpts,my_pe,my_ly,
                                            npes,icx_gpu,icy_gpu,
                                            w_gpu,omega,item);
            }).wait();
            ishmem_barrier_all();
            q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                zouHeBCSHMEMKernel(distr,distrAdv,stencilOpPt,
                                        haloBuff,msgSize,nbr_upper,
                                        nbr_lower,my_jmin_own,my_jmax_own,
                                        my_jmin_ext,my_jmax_ext,
                                        numpts,my_pe,my_ly,
                                        npes,icx_gpu,icy_gpu,
                                        opp_gpu,uLid,
                                        item);
            }).wait();
            std::swap(distr,distrAdv); 
        } else {
            q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                    collideStreamMPIKernel(distr,distrAdv,stencilOpPt,
                                            msgSize,nbr_upper,
                                            nbr_lower,my_jmin_own,my_jmax_own,
                                            my_jmin_ext,my_jmax_ext,
                                            numpts,myRank,my_ly,
                                            nRanks,icx_gpu,icy_gpu,
                                            w_gpu,omega,item);
            }).wait();
            q.parallel_for(sycl::nd_range<1>(dimGrid[2] * dimBlock[2], dimBlock[2]), [=](sycl::nd_item<1> item) {
                zouHeBCMPIKernel(distr,distrAdv,stencilOpPt,
                                        msgSize,nbr_upper,
                                        nbr_lower,my_jmin_own,my_jmax_own,
                                        my_jmin_ext,my_jmax_ext,
                                        numpts,myRank,my_ly,
                                        nRanks,icx_gpu,icy_gpu,
                                        opp_gpu,uLid,
                                        item);
            }).wait();
            std::swap(distr,distrAdv);
            commHalos(distr,locsToSendGPU,locsToRecvGPU,bufferSend,bufferRecv,
                        nLocsSendToLower,nLocsSendToUpper,nLocsRecvFromLower,
                        nLocsRecvFromUpper,myRank,q);
        }
        
    }
    ishmem_barrier_all();
    double tEnd = shmem_wtime();;

    double tInterval = tEnd - tStart;
    // report max,min,avg runtime among ranks
    double timers[3] = {tInterval,tInterval,tInterval};
    MPI_Allreduce(MPI_IN_PLACE, &timers[0], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &timers[1], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &timers[2], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    timers[2] = timers[2] / nRanks;
    if (myRank == 0)
    {
        std::cout << "Max runtime: " << timers[0] << std::endl;
        std::cout << "Min runtime: " << timers[1] << std::endl;
        std::cout << "Average runtime: " << timers[2] << std::endl;
    }
    q.memcpy(distr_host,distr,_LX_ * my_ly_ext * _STENCILSIZE_*sizeof(double)).wait();
    // write output
    writeOutput(distr_host,icx,icy,myRank,my_jmin_ext,my_jmin_own,my_jmax_own,my_ly,my_ly_ext,commSHMEM);
    ishmem_finalize();
    std::cout << "Simulation completed! " << _LX_ * _LY_ * maxT << " total MFLUP" << std::endl; 
}   
