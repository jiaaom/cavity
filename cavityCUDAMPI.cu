/*
* Multi-GPU Lattice Boltzmann Method (LBM) Implementation for 2D Lid-Driven Cavity Flow
* 
* Author: Aristotle Martin
* 
* This code implements a parallel 2D lid-driven cavity flow simulation using the 
* Lattice Boltzmann Method with CUDA+MPI for multi-GPU execution. The domain is
* decomposed across multiple GPUs using MPI for communication.
*
* PARALLELIZATION STRATEGY:
* - Domain decomposition: 2D grid split into horizontal strips
* - Each MPI rank manages one GPU with a subdomain
* - Halo exchange: Communication of boundary data between neighboring GPUs
* - CUDA kernels: Parallel execution on each GPU for compute-intensive operations
*
* ALGORITHM OVERVIEW (per time step):
* 1. CUDA Collision+Streaming kernels on each GPU subdomain
* 2. MPI halo exchange of boundary data between neighboring GPUs  
* 3. Apply boundary conditions (Zou-He for moving lid, bounce-back for walls)
* 4. Repeat for specified number of time steps
*
* MULTI-GPU COMMUNICATION:
* - MPI non-blocking send/receive for inter-GPU data exchange
* - Host-GPU memory transfers for boundary data
* - Overlapping computation and communication where possible
*
* D2Q9 LATTICE:
* - 9 velocity directions (including rest particle)
* - 2D square lattice with nearest and next-nearest neighbor connections
* - Velocity directions: (0,0), (±1,0), (0,±1), (±1,±1)
*
* GRID DECOMPOSITION EXAMPLE (2 GPUs):
* GPU 0: Rows 0 to 511    (lower subdomain)
* GPU 1: Rows 512 to 1023 (upper subdomain)
* Halo exchange at boundary: row 511 ↔ row 512
*/
#include <cuda.h>
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <sys/time.h>
#include <stddef.h>
#include <vector>

// Multi-GPU Lattice Boltzmann Method constants
#define _STENCILSIZE_ 9    // Number of velocity directions in D2Q9 lattice
#define _LX_ 1024          // Grid size in x-direction (8x larger than serial version!)
#define _LY_ 1024          // Grid size in y-direction (total: 1M lattice sites)
#define _NDIMS_ 2          // Number of spatial dimensions
#define _INVALID_ -1       // Invalid grid index marker
#define _HALO_ 1           // Halo region width for MPI communication

// CUDA constant memory for fast access by all GPU threads
// These values are cached and broadcast to all threads in a warp
__constant__ double omega_gpu;  // BGK relaxation parameter (set at runtime)
__constant__ double uLid_gpu;   // Lid velocity (set at runtime)

// D2Q9 lattice velocity vectors stored in GPU constant memory for fast access
// Layout: 0:(0,0), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,-1), 5:(1,1), 6:(-1,1), 7:(-1,-1), 8:(1,-1)
__constant__ int icx_gpu[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};    // x-components
__constant__ int icy_gpu[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1};    // y-components

// D2Q9 lattice weights stored in GPU constant memory
// w0=4/9 (rest), w1-4=1/9 (cardinal directions), w5-8=1/36 (diagonal directions)
__constant__ double w_gpu[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,
                                           1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};

// Opposite directions for bounce-back boundary conditions
// opp[i] gives the index of the velocity direction opposite to direction i
__constant__ int opp_gpu[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};

/**
 * Convert 2D grid coordinates to linear array index (GPU device function)
 * 
 * This function handles domain decomposition by converting global coordinates
 * to local subdomain coordinates for each GPU.
 * 
 * @param i Global x-coordinate (0 to _LX_-1)
 * @param j Global y-coordinate (0 to _LY_-1)
 * @param my_jmin Starting y-coordinate for this GPU's subdomain
 * @param dimJ Height of this GPU's subdomain
 * @return Linear index for local arrays, or _INVALID_ if out of bounds
 * 
 * DOMAIN DECOMPOSITION:
 * - Global coordinates (i,j) are transformed to local coordinates
 * - Each GPU manages a horizontal strip: relJ = j - my_jmin
 * - Local index: i + _LX_ * relJ (row-major within subdomain)
 */
__device__ int getGridIdx_gpu(int i, int j, int my_jmin, int dimJ) {
    int relI = i;              // x-coordinate remains unchanged
    int relJ = j - my_jmin;    // y-coordinate relative to subdomain start
    
    // Check bounds for this GPU's subdomain
    if (relI < 0 || relJ < 0) {
        return _INVALID_;
    }
    if (relI >= _LX_ || relJ >= dimJ) {
        return _INVALID_;
    }
    
    // Linear index within local subdomain (row-major ordering)
    return i + _LX_ * relJ;
}

/**
 * Convert 2D grid coordinates to linear array index (CPU host function)
 * 
 * Same functionality as GPU version but callable from host code.
 * Used for setting up communication patterns and initial conditions.
 * 
 * @param i Global x-coordinate (0 to _LX_-1)
 * @param j Global y-coordinate (0 to _LY_-1)  
 * @param my_jmin Starting y-coordinate for this MPI rank's subdomain
 * @param dimJ Height of this MPI rank's subdomain
 * @return Linear index for local arrays, or _INVALID_ if out of bounds
 */
int getGridIdx(int i, int j, int my_jmin, int dimJ) {
    int relI = i;              // x-coordinate remains unchanged
    int relJ = j - my_jmin;    // y-coordinate relative to subdomain start
    
    // Check bounds for this MPI rank's subdomain
    if (relI < 0 || relJ < 0) {
        return _INVALID_;
    }
    if (relI >= _LX_ || relJ >= dimJ) {
        return _INVALID_;
    }
    
    // Linear index within local subdomain (row-major ordering)
    return i + _LX_ * relJ;
}

/**
 * MPI Halo Exchange for Multi-GPU Lattice Boltzmann Communication
 * 
 * This function handles the critical inter-GPU communication required for
 * domain decomposition in multi-GPU LBM. It exchanges boundary data (halos)
 * between neighboring GPU subdomains using non-blocking MPI communication.
 * 
 * COMMUNICATION PATTERN:
 * - Each GPU needs boundary data from its vertical neighbors
 * - GPU 0 sends to/receives from GPU 1 (upper neighbor)
 * - GPU 1 sends to/receives from GPU 0 (lower neighbor)  
 * - Non-blocking MPI allows overlapping communication with computation
 * 
 * HALO EXCHANGE PROCESS:
 * 1. Pack boundary data from GPU into host send buffers
 * 2. Start non-blocking MPI send/receive operations
 * 3. Wait for all communication to complete
 * 4. Unpack received data back into GPU arrays
 * 
 * @param distr Distribution functions array on host (for boundary data)
 * @param locsToSendGPU Array indices of data to send to neighbors
 * @param locsToRecvGPU Array indices where received data should be stored
 * @param bufferSend Host buffer for outgoing boundary data
 * @param bufferRecv Host buffer for incoming boundary data  
 * @param nLocsSendToLower Number of boundary points to send to lower GPU
 * @param nLocsSendToUpper Number of boundary points to send to upper GPU
 * @param nLocsRecvFromLower Number of boundary points to receive from lower GPU
 * @param nLocsRecvFromUpper Number of boundary points to receive from upper GPU
 * @param myRank Current MPI rank (GPU ID)
 */
void commHalos(double* distr, int* locsToSendGPU, int* locsToRecvGPU, double* bufferSend, double* bufferRecv, int nLocsSendToLower, int nLocsSendToUpper, int nLocsRecvFromLower, int nLocsRecvFromUpper, int myRank) {
    
    // STEP 1: Pack boundary data into send buffer
    int commLength = nLocsSendToUpper + nLocsSendToLower;
    for (int sendIdx = 0; sendIdx < commLength; sendIdx++) {
        int sendLoc = locsToSendGPU[sendIdx];  // Index in local array
        bufferSend[sendIdx] = distr[sendLoc];  // Copy data to send buffer
    }
    
    // STEP 2: Setup non-blocking MPI communication
    MPI_Request* sendRequest = new MPI_Request[2];  // Max 2 sends (lower+upper neighbors)
    MPI_Request* recvRequest = new MPI_Request[2];  // Max 2 receives (lower+upper neighbors)
    int tag = 44;          // MPI message tag for identification
    int sendCount = 0;     // Number of active send requests
    int recvCount = 0;     // Number of active receive requests
    
    // STEP 3: Communicate with lower neighbor (rank-1)
    if (nLocsRecvFromLower > 0) {
        // Post non-blocking receive from lower neighbor
        MPI_Irecv(&bufferRecv[0], nLocsRecvFromLower, MPI_DOUBLE, myRank-1, tag, MPI_COMM_WORLD, recvRequest+recvCount);
        recvCount++;
        // Post non-blocking send to lower neighbor
        MPI_Isend(&bufferSend[0], nLocsSendToLower, MPI_DOUBLE, myRank-1, tag, MPI_COMM_WORLD, sendRequest+sendCount);
        sendCount++;
    }
    
    // STEP 4: Communicate with upper neighbor (rank+1)
    if (nLocsRecvFromUpper > 0) {
        int recvOffset = nLocsRecvFromLower;   // Offset in receive buffer
        int sendOffset = nLocsSendToLower;     // Offset in send buffer
        // Post non-blocking receive from upper neighbor
        MPI_Irecv(&bufferRecv[recvOffset],nLocsRecvFromUpper,MPI_DOUBLE,myRank+1,tag,MPI_COMM_WORLD,recvRequest+recvCount);
        recvCount++;
        // Post non-blocking send to upper neighbor
        MPI_Isend(&bufferSend[sendOffset],nLocsSendToUpper,MPI_DOUBLE,myRank+1,tag,MPI_COMM_WORLD,sendRequest+sendCount);
        sendCount++;
    }
    
    // STEP 5: Wait for all communication to complete
    MPI_Waitall(sendCount, sendRequest, MPI_STATUSES_IGNORE);
    MPI_Waitall(recvCount, recvRequest, MPI_STATUSES_IGNORE);

    // STEP 6: Unpack received data into local arrays
    for (int recvIdx = 0; recvIdx < commLength; recvIdx++) {
        int recvLoc = locsToRecvGPU[recvIdx];     // Where to store in local array
        distr[recvLoc] = bufferRecv[recvIdx];     // Copy from receive buffer
    }
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

/**
 * CUDA Kernel: Initialize Fluid Distribution Functions and Streaming Table
 * 
 * This kernel sets up the initial conditions for the LBM simulation on each GPU.
 * It initializes distribution functions to equilibrium values and precomputes
 * streaming destinations for efficient memory access during simulation.
 * 
 * PARALLEL EXECUTION:
 * - Each CUDA thread processes multiple lattice sites using grid-stride loop
 * - Threads cooperate to initialize entire GPU subdomain
 * - Memory coalescing optimized for efficient GPU memory access
 * 
 * STREAMING TABLE SETUP:
 * - For each site and direction, precompute where particles will stream
 * - Handle boundary conditions: bounce-back for walls, halo exchange for MPI boundaries
 * - This avoids conditional logic in the time-critical collision-streaming kernel
 * 
 * @param distr Distribution functions array (initialized to equilibrium)
 * @param distrAdv Workspace array for time-stepping (initialized to zero)
 * @param stencilOpPt Precomputed streaming destinations array
 * @param my_jmin_ext Starting y-coordinate for this GPU's extended domain (with halos)
 * @param my_ly_ext Height of this GPU's extended domain (with halos)
 * @param numpts Number of lattice sites for this GPU to process
 */
__global__ void initializeFluidKernel(double* distr, double* distrAdv, int* stencilOpPt,
                            int my_jmin_ext, int my_ly_ext, int numpts) {
    
    // CUDA thread indexing: each thread gets a unique ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Grid-stride loop: each thread processes multiple lattice sites
    // This pattern ensures good performance regardless of grid size
    for (int ii = tid; ii < numpts; ii += gridDim.x * blockDim.x) {
        // Convert linear index to 2D coordinates within GPU subdomain
        int myI = ii % _LX_;                    // x-coordinate (0 to _LX_-1)
        int myJ = ii / _LX_ + my_jmin_ext;      // Global y-coordinate
        
        // Initialize all 9 velocity directions for this lattice site
        for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
            // Calculate where particles in direction iNbr will stream to
            int nbrI = myI + icx_gpu[iNbr];     // Destination x-coordinate  
            int nbrJ = myJ + icy_gpu[iNbr];     // Destination y-coordinate
            int nbrIJ = getGridIdx_gpu(nbrI, nbrJ, my_jmin_ext, my_ly_ext);
            
            if (nbrIJ < 0) { 
                // CASE 1: Destination is outside this GPU's domain (MPI boundary)
                // Apply bounce-back until halo exchange provides correct data
                stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii * _STENCILSIZE_ + opp_gpu[iNbr]; 
            }
            else {
                // Check if destination is a solid wall (global boundary)
                if (nbrI < 0 || nbrI >= _LX_ || nbrJ < 0 || nbrJ >= _LY_) {
                    // CASE 2: Solid wall boundary - apply bounce-back
                    stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii * _STENCILSIZE_ + opp_gpu[iNbr];
                }
                else {
                    // CASE 3: Normal fluid streaming to neighboring site
                    stencilOpPt[ii*_STENCILSIZE_+iNbr] = nbrIJ * _STENCILSIZE_ + iNbr;
                }
            }
            
            // Initialize distribution functions to equilibrium (fluid at rest, ρ=1)
            distr[ii*_STENCILSIZE_+iNbr] = w_gpu[iNbr];     // f_i = w_i
            distrAdv[ii*_STENCILSIZE_+iNbr] = 0.0;          // Clear workspace array
        }
    }
}

/**
 * CUDA Kernel: Collision and Streaming Step - Core Multi-GPU LBM Algorithm
 * 
 * This is the computationally intensive kernel that performs the heart of the
 * Lattice Boltzmann Method on each GPU. It combines the collision and streaming
 * operations for maximum GPU performance and memory efficiency.
 * 
 * ALGORITHM OVERVIEW:
 * 1. Calculate macroscopic quantities (density, velocity) from distribution functions
 * 2. Compute Maxwell-Boltzmann equilibrium distribution 
 * 3. Apply BGK collision operator to relax toward equilibrium
 * 4. Stream particles to neighboring sites using precomputed destinations
 * 
 * GPU OPTIMIZATION:
 * - Grid-stride loop for optimal thread utilization
 * - Constant memory access for lattice parameters (cached, broadcasted)
 * - Coalesced memory access patterns for distribution functions
 * - Minimal divergent branching for maximum warp efficiency
 * 
 * MULTI-GPU CONSIDERATIONS:
 * - Each GPU processes only its local subdomain
 * - Boundary data exchange handled separately by MPI halo communication
 * - Streaming destinations precomputed to handle inter-GPU boundaries
 * 
 * @param distr Current distribution functions (input)
 * @param distrAdv New distribution functions after collision+streaming (output)
 * @param stencilOpPt Precomputed streaming destinations for efficient memory access
 * @param my_jmin_ext Starting y-coordinate for this GPU's extended domain
 * @param numpts Number of lattice sites for this GPU to process
 */
__global__ void collideStreamKernel(double* distr, double* distrAdv, int* stencilOpPt,
                            int my_jmin_ext, int numpts)
{
    // CUDA thread indexing with grid-stride loop for optimal GPU utilization
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread processes multiple lattice sites to maximize GPU occupancy
    for (int ii = tid; ii < numpts; ii += gridDim.x * blockDim.x) {
        // Convert linear index to 2D coordinates (for debugging/analysis)
        int myI = ii % _LX_;                    // x-coordinate within subdomain
        int myJ = ii / _LX_ + my_jmin_ext;      // Global y-coordinate
        
        // STEP 1: Calculate macroscopic quantities from distribution functions
        double rho = 0.0;                       // Fluid density: ρ = Σ f_i
        double ux = 0.0;                        // x-momentum: ρu_x = Σ f_i * c_ix
        double uy = 0.0;                        // y-momentum: ρu_y = Σ f_i * c_iy
        double distr_local[_STENCILSIZE_];      // Local copy for better performance

        // Sum over all 9 velocity directions 
        for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
            distr_local[iNbr] = distr[ii*_STENCILSIZE_+iNbr];
            rho += distr_local[iNbr];                           // Accumulate density
            ux += distr_local[iNbr] * icx_gpu[iNbr];            // Accumulate x-momentum
            uy += distr_local[iNbr] * icy_gpu[iNbr];            // Accumulate y-momentum
        }

        // Convert momentum densities to velocities: u = (ρu)/ρ
        double orho = 1.0 / rho;                // Compute reciprocal once for efficiency
        ux *= orho;                             // u_x = (ρu_x)/ρ
        uy *= orho;                             // u_y = (ρu_y)/ρ
        double uke = ux * ux + uy * uy;         // Kinetic energy density |u|²

        // STEP 2: Collision and Streaming combined for optimal GPU memory usage
        for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
            // Get precomputed streaming destination (handles boundaries automatically)
            int nbrInd = stencilOpPt[ii*_STENCILSIZE_+iNbr];
            
            // Calculate dot product of lattice velocity with fluid velocity
            double cdotu = icx_gpu[iNbr]*ux + icy_gpu[iNbr]*uy;     // c_i · u
            
            // Maxwell-Boltzmann equilibrium distribution (2nd order expansion)
            // f^eq_i = w_i * ρ * [1 + 3(c_i·u) + 9/2(c_i·u)² - 3/2|u|²]
            double distr_eq = w_gpu[iNbr] * rho * (1.0 + 3.0*cdotu + 4.5*cdotu*cdotu - 1.5*uke);
            
            // BGK collision with streaming: f_new = ω*f^eq + (1-ω)*f_old
            // Stream result to neighboring site determined by stencilOpPt
            distrAdv[nbrInd] = omega_gpu*distr_eq + (1.0-omega_gpu)*distr_local[iNbr];
        }
    }
}

/**
 * CUDA Kernel: Zou-He Boundary Condition for Moving Lid
 * 
 * This kernel applies the Zou-He velocity boundary condition to the moving lid
 * (top boundary at y=0) for the lid-driven cavity flow. Only the GPU that owns
 * the top boundary (typically rank 0) will execute this boundary condition.
 * 
 * ZOU-HE METHOD:
 * - Prescribes velocity at the boundary (horizontal lid motion)
 * - Extrapolates density from the fluid interior
 * - Reconstructs unknown distribution functions to satisfy velocity constraint
 * - Maintains mass conservation and proper momentum transfer
 * 
 * MULTI-GPU CONSIDERATIONS:
 * - Only one GPU (usually rank 0) owns the top boundary
 * - Other GPUs skip this kernel since myJ ≠ 0 for their domains
 * - Ensures consistent boundary conditions across domain decomposition
 * 
 * @param distr Input distribution functions (unused in this kernel)
 * @param distrAdv Modified distribution functions with boundary conditions applied
 * @param stencilOpPt Precomputed streaming table (unused in this kernel)
 * @param my_jmin_ext Starting y-coordinate for this GPU's extended domain
 * @param numpts Number of lattice sites for this GPU to process
 */
__global__ void zouHeBCKernel(double* distr, double* distrAdv, int* stencilOpPt,
                    int my_jmin_ext, int numpts) {
    // CUDA thread indexing
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Grid-stride loop over all lattice sites in this GPU's domain
    for (int ii = tid; ii < numpts; ii += gridDim.x * blockDim.x) {
        int myI = ii % _LX_;                    // x-coordinate within subdomain
        int myJ = ii / _LX_ + my_jmin_ext;      // Global y-coordinate
        
        if (myJ == 0) { // Apply boundary condition only at the lid (y=0)
            // Prescribed lid velocities
            double ux = uLid_gpu;    // Horizontal lid velocity (from constant memory)
            double uy = 0.0;         // No vertical velocity at lid
            
            // Extrapolate density from known distribution functions
            // Known: f0, f1, f3, f4, f7, f8 (center, east, west, south, SE, SW)
            double rho = (1.0/(1.0-uy))*(distrAdv[ii*_STENCILSIZE_+0]+   // f0 (center)
                                         distrAdv[ii*_STENCILSIZE_+1]+   // f1 (east)
                                         distrAdv[ii*_STENCILSIZE_+3]+   // f3 (west)
                                         2*(distrAdv[ii*_STENCILSIZE_+4]+ // f4 (south)
                                            distrAdv[ii*_STENCILSIZE_+7]+ // f7 (SW)
                                            distrAdv[ii*_STENCILSIZE_+8])); // f8 (SE)

            // Reconstruct unknown distributions using Zou-He formulas
            // f2 = f4 + (2/3)*ρ*uy (north direction)
            distrAdv[ii*_STENCILSIZE_+2] = distrAdv[ii*_STENCILSIZE_+4] + (2.0/3.0)*rho*uy;
            
            // f5 = f7 - (1/2)*(f1-f3) + (1/2)*ρ*ux - (1/6)*ρ*uy (NE direction)
            distrAdv[ii*_STENCILSIZE_+5] = distrAdv[ii*_STENCILSIZE_+7] - 
                                           (1.0/2.0)*(distrAdv[ii*_STENCILSIZE_+1] - distrAdv[ii*_STENCILSIZE_+3]) + 
                                           (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
            
            // f6 = f8 + (1/2)*(f1-f3) - (1/2)*ρ*ux - (1/6)*ρ*uy (NW direction)  
            distrAdv[ii*_STENCILSIZE_+6] = distrAdv[ii*_STENCILSIZE_+8] + 
                                           (1.0/2.0)*(distrAdv[ii*_STENCILSIZE_+1] - distrAdv[ii*_STENCILSIZE_+3]) - 
                                           (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
        }
    }
}

void writeOutput(double* distr, int* icx, int* icy, int myRank, int my_jmin_ext, int my_jmin_own, int my_jmax_own, int my_ly_ext) {
    std::string file_name = "out_" + std::to_string(myRank) + ".txt";
    std::ofstream out_file(file_name);

    for (int idxJ=my_jmin_own; idxJ<=my_jmax_own; idxJ++) {
        for (int idxI=0; idxI<_LX_; idxI++) {
            int idxIJ = getGridIdx(idxI,idxJ,my_jmin_ext,my_ly_ext);
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

/**
 * MAIN PROGRAM: Multi-GPU Lattice Boltzmann Method for Lid-Driven Cavity Flow
 * 
 * This is the main driver function that orchestrates the multi-GPU LBM simulation.
 * It handles MPI initialization, domain decomposition, GPU memory management,
 * and the main simulation loop with inter-GPU communication.
 * 
 * EXECUTION OVERVIEW:
 * 1. Initialize MPI and determine rank/size
 * 2. Decompose domain across GPUs (horizontal strip decomposition)
 * 3. Set up GPU memory and communication buffers
 * 4. Run simulation loop: collision-streaming + boundary conditions + halo exchange
 * 5. Write results and finalize MPI
 * 
 * DOMAIN DECOMPOSITION STRATEGY:
 * - 1024×1024 total grid divided into horizontal strips
 * - Each GPU gets ~512 rows (for 2 GPUs)
 * - Includes halo regions for boundary data exchange
 * - GPU 0: rows 0-511, GPU 1: rows 512-1023
 * 
 * MEMORY MANAGEMENT:
 * - Uses CUDA unified memory for easy CPU-GPU data sharing
 * - Distribution functions: ~37MB per GPU (1024×512×9 × sizeof(double))
 * - Communication buffers for MPI halo exchange
 */
int main(int argc, char** argv) {
    // =============================================================================
    // MPI INITIALIZATION AND RANK SETUP
    // =============================================================================
    MPI_Init(&argc, &argv);
    int myRank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    // Get this process's rank (GPU ID)
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);   // Get total number of processes (GPUs)

    // =============================================================================
    // DOMAIN DECOMPOSITION: Divide 1024×1024 grid across GPUs
    // =============================================================================
    int my_jmin_own, my_jmax_own;  // This GPU's owned y-range (no overlap)
    int my_jmin_ext, my_jmax_ext;  // This GPU's extended y-range (with halos)
    
    // Calculate owned domain boundaries for horizontal strip decomposition
    my_jmin_own = round(_LY_ / nRanks) * myRank;           // Start of owned region
    my_jmax_own = round(_LY_ / nRanks) * (myRank + 1) - 1; // End of owned region
    
    // Handle boundary cases and ensure proper coverage
    if (my_jmin_own < 0) my_jmin_own = 0;                  // Clamp to valid range
    if (my_jmax_own > _LY_ - 1) my_jmax_own = _LY_ - 1;    // Clamp to valid range
    if (myRank == nRanks - 1) my_jmax_own = _LY_ - 1;      // Last rank gets remainder
    
    // Calculate extended domain with halo regions for communication
    my_jmin_ext = my_jmin_own - _HALO_ >= 0 ? my_jmin_own - _HALO_ : 0;
    my_jmax_ext = my_jmax_own + _HALO_ < _LY_ ? my_jmax_own + _HALO_ : _LY_ - 1;

    // Calculate domain dimensions and neighbor ranks
    int my_ly = my_jmax_own - my_jmin_own + 1;      // Height of owned domain
    int my_ly_ext = my_jmax_ext - my_jmin_ext + 1;  // Height of extended domain (with halos)
    int msgSize = _LX_ * _STENCILSIZE_;              // Size of boundary messages
    int nbr_upper = myRank + 1 < nRanks ? myRank + 1 : 0;        // Upper neighbor rank
    int nbr_lower = myRank - 1 >= 0 ? myRank - 1 : nRanks - 1;   // Lower neighbor rank

    // Debug output: show domain decomposition
    std::cout << "Rank: " << myRank << ", nbr_upper = " << nbr_upper << ", nbr_lower = " << nbr_lower << std::endl;
    std::cout << "Rank: " << myRank << ", my_jmin_own = " << my_jmin_own << ", my_jmax_own = " << my_jmax_own << std::endl;
    std::cout << "Rank: " << myRank << ", my_jmin_ext = " << my_jmin_ext << ", my_jmax_ext = " << my_jmax_ext << std::endl;

    // =============================================================================
    // SIMULATION PARAMETERS (Same as serial version but larger grid)
    // =============================================================================
    int maxT = 10000;           // Total number of time steps to simulate

    double uLid = 0.05;         // Horizontal velocity of moving lid (lattice units)
    double Re = 100.0;          // Reynolds number = U*L/ν (dimensionless)

    double nu = uLid * _LX_ / Re;           // Kinematic viscosity from Re definition
    double omega = 1.0 / (3.0*nu + 0.5);   // BGK relaxation parameter: ω = 1/(3ν + 0.5)

    // =============================================================================
    // GPU MEMORY ALLOCATION AND SETUP
    // =============================================================================
    double* distr;        // Current distribution functions on GPU
    double* distrAdv;     // Next time step distribution functions on GPU
    int* stencilOpPt;     // Precomputed streaming destinations on GPU
    
    // D2Q9 lattice constants (host copies for setup)
    int icx[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};        // x-velocity components
    int icy[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1};        // y-velocity components
    int opp[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};           // Opposite directions
    double w[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,
                              1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0}; // Lattice weights

    // Allocate GPU memory using CUDA unified memory for easy access from both CPU and GPU
    // Memory size per GPU: ~37MB for distribution functions
    cudaMallocManaged((void**)&distr, _LX_ * my_ly_ext * _STENCILSIZE_ * sizeof(double));
    cudaMallocManaged((void**)&distrAdv, _LX_ * my_ly_ext * _STENCILSIZE_ * sizeof(double));
    cudaMallocManaged((void**)&stencilOpPt, _LX_ * my_ly_ext * _STENCILSIZE_ * sizeof(int));
    
    // Copy simulation parameters to GPU constant memory for fast access
    cudaMemcpyToSymbol(omega_gpu, &omega, sizeof(double));   // Relaxation parameter
    cudaMemcpyToSymbol(uLid_gpu, &uLid, sizeof(double));     // Lid velocity
    
    int numpts = my_ly_ext * _LX_;  // Total number of lattice sites for this GPU

    // MPI communication stuff
    std::vector<int> locsToSend;
    std::vector<int> locsToSendAoS;
    std::vector<int> locsToRecv;
    std::vector<int> locsToRecvAoS;
    std::vector<int> locSendCounts(2,0);
    std::vector<int> locRecvCounts(2,0);
    setupSendLocs(locsToSend,locsToSendAoS,locSendCounts,myRank,nRanks,
                    my_jmin_own,my_jmax_own,my_jmin_ext,my_jmax_ext,my_ly_ext,
                    &icx[0],&icy[0]);
    exchangeLocSizes(locSendCounts,locRecvCounts,myRank,nRanks);
    exchangeLocAoS(locsToSendAoS,locsToRecvAoS,locSendCounts,locRecvCounts,myRank);
    setupRecvLocs(locsToRecv,locsToRecvAoS,locRecvCounts,my_jmin_ext,my_jmin_own,my_ly_ext);
    // message buffers
    int nLocsRecvFromLower = locRecvCounts[0];
    int nLocsRecvFromUpper = locRecvCounts[1];
    int nTotRecvLocs = nLocsRecvFromLower + nLocsRecvFromUpper;
    int nLocsSendToLower = locSendCounts[0];
    int nLocsSendToUpper = locSendCounts[1];
    int nTotSendLocs = nLocsSendToLower + nLocsSendToUpper;
    assert(nTotSendLocs == nTotRecvLocs);
    int* locsToSendGPU;
    int* locsToRecvGPU;
    double* bufferSend;
    double* bufferRecv;
    cudaMallocManaged((void**)&locsToSendGPU,nTotSendLocs*sizeof(int));
    cudaMallocManaged((void**)&locsToRecvGPU,nTotRecvLocs*sizeof(int));
    cudaMallocManaged((void**)&bufferSend,nTotSendLocs*sizeof(double));
    cudaMallocManaged((void**)&bufferRecv,nTotRecvLocs*sizeof(double));
    cudaMemcpy(locsToSendGPU,locsToSend.data(),nTotSendLocs*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(locsToRecvGPU,locsToRecv.data(),nTotRecvLocs*sizeof(int),cudaMemcpyHostToDevice);

    // =============================================================================
    // CUDA KERNEL LAUNCH CONFIGURATION
    // =============================================================================
    const dim3 dimBlock(128, 1, 1);  // 128 threads per block (optimal for most GPUs)
    const int gridDimX = ceil(numpts / (double)dimBlock.x);  // Enough blocks to cover all lattice sites
    const int gridDimY = 1;           // 1D grid is sufficient for our memory access pattern
    dim3 dimGrid(gridDimX, gridDimY, 1);

    // =============================================================================
    // MAIN SIMULATION LOOP: Multi-GPU Lattice Boltzmann Method
    // =============================================================================
    double tStart = MPI_Wtime();  // Start timing the simulation
    
    // INITIALIZATION: Set up distribution functions and streaming table
    initializeFluidKernel<<<dimGrid, dimBlock>>>(distr, distrAdv, stencilOpPt, 
                                                my_jmin_ext, my_ly_ext, numpts);
    
    // MAIN TIME-STEPPING LOOP
    for (int t = 0; t < maxT; t++) {
        // STEP 1: Collision and Streaming (core LBM computation)
        // Each GPU processes its subdomain in parallel
        collideStreamKernel<<<dimGrid, dimBlock>>>(distr, distrAdv, stencilOpPt, 
                                                  my_jmin_ext, numpts);
        
        // STEP 2: Apply Boundary Conditions
        // Only GPU 0 applies lid boundary condition (y=0), others skip
        zouHeBCKernel<<<dimGrid, dimBlock>>>(distr, distrAdv, stencilOpPt, 
                                           my_jmin_ext, numpts);
        
        // STEP 3: Swap arrays for next iteration (ping-pong scheme)
        std::swap(distr, distrAdv);
        
        // STEP 4: Inter-GPU Communication (MPI Halo Exchange)
        // Exchange boundary data between neighboring GPUs
        commHalos(distr, locsToSendGPU, locsToRecvGPU, bufferSend, bufferRecv,
                  nLocsSendToLower, nLocsSendToUpper, nLocsRecvFromLower, nLocsRecvFromUpper, myRank);
    }
    
    // Synchronize all GPUs and stop timing
    MPI_Barrier(MPI_COMM_WORLD);
    double tEnd = MPI_Wtime();

    // =============================================================================
    // PERFORMANCE REPORTING AND RESULTS OUTPUT  
    // =============================================================================
    double tInterval = tEnd - tStart;  // Total simulation time for this GPU
    
    // Gather performance statistics across all GPUs using MPI reductions
    double timers[3] = {tInterval, tInterval, tInterval};  // max, min, avg
    MPI_Allreduce(MPI_IN_PLACE, &timers[0], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);  // Max time
    MPI_Allreduce(MPI_IN_PLACE, &timers[1], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);  // Min time  
    MPI_Allreduce(MPI_IN_PLACE, &timers[2], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // Sum for average
    timers[2] = timers[2] / nRanks;  // Calculate average runtime across all GPUs
    
    // Print performance results (only from rank 0 to avoid duplicate output)
    if (myRank == 0) {
        std::cout << "Max runtime: " << timers[0] << std::endl;      // Slowest GPU
        std::cout << "Min runtime: " << timers[1] << std::endl;      // Fastest GPU
        std::cout << "Average runtime: " << timers[2] << std::endl;  // Average across GPUs
    }
    
    // Each GPU writes its portion of results to separate output file
    // Output files: out_0.txt, out_1.txt, etc.
    writeOutput(distr, icx, icy, myRank, my_jmin_ext, my_jmin_own, my_jmax_own, my_ly_ext);
    
    // =============================================================================
    // CLEANUP AND MPI FINALIZATION
    // =============================================================================
    MPI_Finalize();  // Clean shutdown of MPI environment
    std::cout << "Simulation completed! " << _LX_ * _LY_ * maxT << " total MFLUP" << std::endl; 
    return 0;  // Successful program termination
}   