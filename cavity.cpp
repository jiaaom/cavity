/*
* Lattice Boltzmann Method (LBM) Implementation for 2D Lid-Driven Cavity Flow
* 
* Author: Aristotle Martin
* 
* This code implements a serial 2D lid-driven cavity flow simulation using the 
* Lattice Boltzmann Method with a D2Q9 lattice structure. The cavity has a 
* moving top lid and stationary walls on the other three sides.
*
* Algorithm Overview:
* 1. Initialize fluid distributions to equilibrium values
* 2. For each time step:
*    a. Collision step: Relax distributions towards local equilibrium
*    b. Streaming step: Move distributions to neighboring lattice sites
*    c. Apply boundary conditions (Zou-He for moving lid, bounce-back for walls)
* 3. Output macroscopic quantities (velocity, density)
*
* D2Q9 Lattice:
* - 9 velocity directions (including rest particle)
* - 2D square lattice with nearest and next-nearest neighbor connections
* - Velocity directions: (0,0), (±1,0), (0,±1), (±1,±1)
*/
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <vector>
#include <sys/time.h>

// Lattice Boltzmann Method constants
#define _STENCILSIZE_ 9    // Number of velocity directions in D2Q9 lattice
#define _LX_ 128           // Grid size in x-direction
#define _LY_ 128           // Grid size in y-direction  
#define _NDIMS_ 2          // Number of spatial dimensions
#define _INVALID_ -1       // Invalid grid index marker

using namespace std;

/**
 * Convert 2D grid coordinates to linear array index
 * 
 * @param i x-coordinate (0 to _LX_-1)
 * @param j y-coordinate (0 to _LY_-1)
 * @return Linear index for accessing arrays, or _INVALID_ if out of bounds
 * 
 * Uses row-major ordering: index = i + _LX_ * j
 */
int getGridIdx(int i, int j) {
    if (i < 0 || i >= _LX_ || j < 0 || j >= _LY_) {
        return _INVALID_;
    }
    return i + _LX_ * j;
}

/**
 * Write simulation results to output file
 * 
 * Calculates macroscopic quantities (density, velocity) from distribution functions
 * and writes them to "out.txt" in CSV format: x, y, ux, uy, rho
 * 
 * @param distr Distribution functions for all lattice sites and directions
 * @param icx Lattice velocity components in x-direction
 * @param icy Lattice velocity components in y-direction
 */
void writeOutput(vector<double>& distr, int* icx, int* icy) {
    std::string file_name = "out.txt";
    std::ofstream out_file(file_name);

    // Loop through all grid points
    for (int idxI=0; idxI<_LX_; idxI++) {
        for (int idxJ=0; idxJ<_LY_; idxJ++) {
            int idxIJ = getGridIdx(idxI,idxJ);
            
            // Calculate macroscopic quantities from distribution functions
            double rho = 0.0;  // Density
            double ux = 0.0;   // x-velocity
            double uy = 0.0;   // y-velocity
            double distr_local[_STENCILSIZE_];
            
            // Sum over all velocity directions
            for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
                distr_local[iNbr] = distr[idxIJ*_STENCILSIZE_+iNbr];
                rho += distr_local[iNbr];                          // ρ = Σ f_i
                ux += distr_local[iNbr] * icx[iNbr];               // ρu_x = Σ f_i * c_ix
                uy += distr_local[iNbr] * icy[iNbr];               // ρu_y = Σ f_i * c_iy
            }
            
            // Convert momentum to velocity: u = (ρu)/ρ
            double orho = 1.0 / rho;
            ux *= orho;
            uy *= orho;
            
            // Write to file: x, y, ux, uy, rho
            out_file << std::setprecision(16) << idxI << ", " << idxJ << ", " 
                     << ux << ", " << uy << ", " << rho << std::endl;
        }
    }

    out_file.close();
}

/**
 * Apply Zou-He boundary condition for moving lid
 * 
 * Implements the Zou-He method for velocity boundary conditions on the top wall (lid).
 * The lid moves with prescribed horizontal velocity uLid and zero vertical velocity.
 * This method extrapolates the density from the bulk fluid and reconstructs
 * the unknown distribution functions to satisfy the velocity constraint.
 * 
 * D2Q9 velocity indices:
 * 0:(0,0), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,-1), 5:(1,1), 6:(-1,1), 7:(-1,-1), 8:(1,-1)
 * 
 * For top boundary (myJ=0), unknown distributions are: 2, 5, 6 (pointing upward)
 * Known distributions are: 0,1,3,4,7,8 (rest and downward/horizontal)
 * 
 * @param distr Distribution functions to modify
 * @param uLid Horizontal velocity of the moving lid
 */
void zouHeBC(vector<double>& distr, double uLid) {
    int myJ = 0; // y-coordinate of the moving lid (top boundary)
    
    for (int myI = 0; myI < _LX_; myI++) {
        int idxIJ = getGridIdx(myI, myJ);

        // Prescribed boundary velocities
        double ux = uLid;  // Horizontal lid velocity
        double uy = 0.0;   // No vertical velocity at lid
        
        // Extrapolate density from known distributions
        // ρ = (1/(1-uy)) * (f0 + f1 + f3 + 2*(f4 + f7 + f8))
        double rho = (1.0/(1.0-uy))*(distr[idxIJ*_STENCILSIZE_+0]+
                                     distr[idxIJ*_STENCILSIZE_+1]+
                                     distr[idxIJ*_STENCILSIZE_+3]+
                                     2*(distr[idxIJ*_STENCILSIZE_+4]+
                                        distr[idxIJ*_STENCILSIZE_+7]+
                                        distr[idxIJ*_STENCILSIZE_+8]));

        // Reconstruct unknown distributions using Zou-He formulas
        // f2 = f4 + (2/3)*ρ*uy  (upward, direction 2)
        distr[idxIJ*_STENCILSIZE_+2] = distr[idxIJ*_STENCILSIZE_+4] + (2.0/3.0)*rho*uy;
        
        // f5 = f7 - (1/2)*(f1-f3) + (1/2)*ρ*ux - (1/6)*ρ*uy  (diagonal up-right, direction 5)
        distr[idxIJ*_STENCILSIZE_+5] = distr[idxIJ*_STENCILSIZE_+7] - 
                                       (1.0/2.0)*(distr[idxIJ*_STENCILSIZE_+1] - distr[idxIJ*_STENCILSIZE_+3]) + 
                                       (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
        
        // f6 = f8 + (1/2)*(f1-f3) - (1/2)*ρ*ux - (1/6)*ρ*uy  (diagonal up-left, direction 6)
        distr[idxIJ*_STENCILSIZE_+6] = distr[idxIJ*_STENCILSIZE_+8] + 
                                       (1.0/2.0)*(distr[idxIJ*_STENCILSIZE_+1] - distr[idxIJ*_STENCILSIZE_+3]) - 
                                       (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
    }
}

/**
 * Collision and Streaming Step - Core of the Lattice Boltzmann Method
 * 
 * This function performs the heart of the LBM algorithm in two conceptual steps:
 * 
 * 1. COLLISION: Particles at each lattice site relax towards local equilibrium
 *    - This models viscous effects and drives the fluid toward equilibrium
 *    - Uses BGK (Bhatnagar-Gross-Krook) approximation for collision operator
 * 
 * 2. STREAMING: Particles move to neighboring lattice sites along velocity vectors
 *    - This models the advection/transport of fluid
 *    - Each distribution function f_i moves in direction c_i
 * 
 * The equilibrium distribution f^eq captures the local Maxwell-Boltzmann distribution
 * expanded to second order in velocity, which recovers the Navier-Stokes equations
 * in the macroscopic limit.
 * 
 * @param distr Current distribution functions (input)
 * @param distrAdv New distribution functions after collision+streaming (output)
 * @param icx x-components of lattice velocity vectors
 * @param icy y-components of lattice velocity vectors  
 * @param w Lattice weights for each velocity direction
 * @param stencilOpPt Precomputed streaming destinations for each site/direction
 * @param omega Relaxation parameter (inverse of relaxation time): ω = 1/(3ν + 0.5)
 */
void collideStream(vector<double>& distr, vector<double>& distrAdv, int* icx, int* icy, double* w, int* stencilOpPt, double omega) {
    // Loop over all lattice sites
    for (int ii=0; ii<_LX_*_LY_; ii++) {
        int myI = ii % _LX_;   // x-coordinate of current site
        int myJ = ii / _LX_;  // y-coordinate of current site

        // STEP 1: Calculate macroscopic variables from distribution functions
        // These represent the actual fluid properties (density, velocity) at this point
        double rho = 0.0;  // Fluid density: ρ = Σ f_i
        double ux = 0.0;   // x-momentum: ρu_x = Σ f_i * c_ix  
        double uy = 0.0;   // y-momentum: ρu_y = Σ f_i * c_iy
        double distr_local[_STENCILSIZE_];

        // Sum over all 9 velocity directions
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            distr_local[iNbr] = distr[ii*_STENCILSIZE_+iNbr];
            rho += distr_local[iNbr];                    // Total mass density
            ux += distr_local[iNbr] * icx[iNbr];         // x-momentum density
            uy += distr_local[iNbr] * icy[iNbr];         // y-momentum density
        }

        // Convert momentum densities to velocities: u = (ρu)/ρ
        double orho = 1.0 / rho;
        ux *= orho;
        uy *= orho;
        double uke = ux * ux + uy * uy;  // Kinetic energy density |u|²

        // STEP 2: Collision and Streaming combined
        // For each velocity direction, compute new distribution function
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            // Determine where this particle will stream to
            int nbrInd = stencilOpPt[ii*_STENCILSIZE_+iNbr];
            
            // Dot product of lattice velocity with fluid velocity: c_i · u
            double cdotu = icx[iNbr]*ux + icy[iNbr]*uy;
            
            // Maxwell-Boltzmann equilibrium distribution (2nd order expansion)
            // f^eq_i = w_i * ρ * [1 + 3(c_i·u) + 9/2(c_i·u)² - 3/2|u|²]
            double distr_eq = w[iNbr] * rho * (1.0 + 3.0*cdotu + 4.5*cdotu*cdotu - 1.5*uke);
            
            // BGK collision: f_new = ω*f^eq + (1-ω)*f_old
            // ω=1: instantaneous relaxation to equilibrium (inviscid)
            // ω→0: no relaxation (infinite viscosity)
            // Streaming: place result at neighboring site determined by stencilOpPt
            distrAdv[nbrInd] = omega*distr_eq + (1.0-omega)*distr_local[iNbr];
        }
    }
}

/**
 * Setup Streaming Adjacency Table
 * 
 * This function precomputes where each distribution function will stream to,
 * implementing both fluid streaming and wall boundary conditions.
 * 
 * STREAMING CONCEPT:
 * - Each f_i at site (x,y) moves to site (x + c_ix, y + c_iy)
 * - For interior fluid: particles simply move to the neighboring site
 * - For solid walls: particles bounce back in the opposite direction (no-slip condition)
 * 
 * BOUNDARY CONDITIONS:
 * - Bounce-back rule: when a particle hits a wall, it reverses direction
 * - This enforces zero velocity at walls (no-slip condition)
 * - Example: f_1 (moving right) hitting right wall becomes f_3 (moving left)
 * 
 * The cavity has solid walls on bottom, left, and right sides.
 * The top boundary (lid) is handled separately by the Zou-He method.
 * 
 * @param stencilOpPt Output array: destination index for each (site, direction) pair
 * @param icx Lattice velocity x-components
 * @param icy Lattice velocity y-components  
 * @param opp Opposite direction indices for bounce-back
 */
void setupAdjacency(vector<int>& stencilOpPt, int* icx, int* icy, int* opp) {
    // Loop over all lattice sites
    for (int ii=0; ii<_LX_*_LY_; ii++) {
        int myI = ii % _LX_;   // Current x-coordinate
        int myJ = ii / _LX_;  // Current y-coordinate
        
        // For each velocity direction from this site
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            // Calculate destination coordinates after streaming
            int nbrI = myI + icx[iNbr];  // New x-coordinate
            int nbrJ = myJ + icy[iNbr];  // New y-coordinate
            int nbrIJ = getGridIdx(nbrI, nbrJ);  // Linear index of destination
            
            if (nbrIJ < 0) {
                // BOUNDARY CONDITION: Destination is outside domain (solid wall)
                // Apply bounce-back: particle returns to origin in opposite direction
                // Example: f_1 (east) hitting east wall becomes f_3 (west) at same site
                stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii * _STENCILSIZE_ + opp[iNbr];
            }
            else {
                // FLUID STREAMING: Normal streaming to neighboring lattice site
                // Particle at (ii, direction iNbr) goes to (nbrIJ, same direction)
                stencilOpPt[ii*_STENCILSIZE_+iNbr] = nbrIJ * _STENCILSIZE_ + iNbr;
            }
        }
    }
}

/**
 * Initialize Fluid Distribution Functions
 * 
 * Sets up the initial state of the fluid by initializing all distribution
 * functions to their equilibrium values for a fluid at rest with unit density.
 * 
 * PHYSICAL MEANING:
 * - At t=0, the fluid is at rest (u=0) with uniform density (ρ=1)
 * - The equilibrium distribution for this state is simply f_i^eq = w_i * ρ = w_i
 * - This represents a quiescent fluid before the lid starts moving
 * 
 * INITIALIZATION STRATEGY:
 * - Each f_i gets the lattice weight w_i (ensures ρ=1, u=0)
 * - This is the Maxwell-Boltzmann equilibrium for a stationary fluid
 * - The simulation will evolve from this initial condition
 * 
 * @param distr Primary distribution array to initialize
 * @param distrAdv Secondary distribution array (initialized to zero)
 * @param w Lattice weights for D2Q9 stencil
 */
void initializeFluid(vector<double>& distr, vector<double>& distrAdv, double* w) {
    // Initialize all lattice sites
    for (int ii=0; ii<_LX_*_LY_; ii++) {
        // Initialize all velocity directions at this site
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            // Set to equilibrium for fluid at rest: f_i = w_i * ρ = w_i (ρ=1)
            distr[ii*_STENCILSIZE_+iNbr] = w[iNbr];
            // Initialize workspace array to zero
            distrAdv[ii*_STENCILSIZE_+iNbr] = 0.0;
        }
    }
}

/**
 * Setup Grid Coordinate Mapping
 * 
 * Creates a lookup table that maps linear indices back to (x,y) coordinates.
 * This is primarily used for debugging and analysis purposes.
 * 
 * COORDINATE SYSTEM:
 * - Origin (0,0) is at bottom-left corner
 * - x increases rightward (0 to _LX_-1)  
 * - y increases upward (0 to _LY_-1)
 * - Linear index ii maps to coordinates (ii % _LX_, ii / _LX_)
 * 
 * USAGE:
 * - Allows converting from linear array index back to spatial coordinates
 * - Useful for spatial analysis and visualization of results
 * - Not strictly necessary for the LBM algorithm itself
 * 
 * @param fluidPts Array to store coordinate pairs for each lattice site
 */
void setupGrid(vector<int>& fluidPts) {
    // Loop through all lattice sites
    for (int idxI=0; idxI<_LX_; idxI++) {        // x-coordinate
        for (int idxJ=0; idxJ<_LY_; idxJ++) {    // y-coordinate
            int idxIJ = idxI + _LX_ * idxJ;      // Linear index (row-major)
            
            // Store coordinate pair for this linear index
            fluidPts[idxIJ*_NDIMS_] = idxI;      // x-coordinate
            fluidPts[idxIJ*_NDIMS_+1] = idxJ;    // y-coordinate
        }
    }
}

/**
 * MAIN PROGRAM: 2D Lid-Driven Cavity Flow Simulation
 * 
 * This program simulates fluid flow in a square cavity with a moving top lid.
 * The cavity setup is a classic computational fluid dynamics benchmark:
 * - Top wall (lid): moves horizontally with prescribed velocity
 * - Side and bottom walls: stationary (no-slip condition)
 * - Fluid: initially at rest, driven by lid motion
 * 
 * PHYSICS PARAMETERS:
 * - Reynolds number Re = UL/ν controls the flow regime
 * - Low Re: steady, laminar flow with large recirculation zone  
 * - High Re: unsteady, turbulent flow with multiple vortices
 * - Typical benchmark values: Re = 100, 400, 1000, 3200
 * 
 * LATTICE BOLTZMANN PARAMETERS:
 * - Relaxation time τ = 1/ω controls viscosity
 * - Must satisfy stability condition: 0.5 < τ < 2.0 (i.e., 0.5 < ω < 2.0)
 * - Viscosity: ν = cs²(τ - 0.5) = (1/3)(1/ω - 0.5)
 */
int main() {
    // =============================================================================
    // SIMULATION PARAMETERS
    // =============================================================================
    int maxT = 10000;        // Total number of time steps to simulate
    
    // Physical parameters for cavity flow benchmark
    double uLid = 0.05;      // Horizontal velocity of moving lid (lattice units)
    double Re = 100.0;       // Reynolds number = U*L/ν (dimensionless)
    
    // Lattice Boltzmann derived parameters
    double cs2 = 1.0/3.0;    // Lattice speed of sound squared (cs² = 1/3 for D2Q9)
    double nu = uLid * _LX_ / Re;        // Kinematic viscosity from Re definition
    double omega = 1.0 / (3.0*nu + 0.5); // BGK relaxation parameter: ω = 1/(3ν + 0.5)

    // =============================================================================
    // D2Q9 LATTICE CONSTANTS
    // =============================================================================
    // Discrete velocities for D2Q9 lattice (9 directions including rest)
    // Velocity directions:
    //   6   2   5      (-1,1) (0,1) (1,1)
    //   3   0   1  =>  (-1,0) (0,0) (1,0)  
    //   7   4   8      (-1,-1)(0,-1)(1,-1)
    
    int icx[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};  // x-components of lattice velocities
    int icy[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1};  // y-components of lattice velocities
    
    // Opposite directions for bounce-back boundary conditions
    // opp[i] gives the index of velocity opposite to direction i
    int opp[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};
    
    // Lattice weights for D2Q9 stencil (ensure isotropy and Galilean invariance)
    // w0=4/9 (rest), w1-4=1/9 (cardinal), w5-8=1/36 (diagonal)
    double w[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,
                              1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};

    // =============================================================================
    // MEMORY ALLOCATION
    // =============================================================================
    // Distribution functions: f_i(x,y,t) for each site and velocity direction
    vector<double> distr(_LX_ * _LY_ * _STENCILSIZE_);    // Current time step
    vector<double> distrAdv(_LX_ * _LY_ * _STENCILSIZE_); // Next time step
    
    // Precomputed streaming destinations for efficient memory access
    vector<int> stencilOpPt(_LX_ * _LY_ * _STENCILSIZE_);
    
    // Grid coordinate mapping (for analysis/debugging)
    vector<int> fluidPts(_LX_ * _LY_ * _NDIMS_);

    // =============================================================================
    // INITIALIZATION
    // =============================================================================
    cout << "Initializing 2D Lid-Driven Cavity Flow..." << endl;
    cout << "Grid size: " << _LX_ << "x" << _LY_ << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Lid velocity: " << uLid << " (lattice units)" << endl;
    cout << "Relaxation parameter: " << omega << endl;
    cout << "Time steps: " << maxT << endl << endl;
    
    setupGrid(fluidPts);                              // Setup coordinate mapping
    setupAdjacency(stencilOpPt,icx,icy,opp);          // Precompute streaming table
    initializeFluid(distr, distrAdv, &w[0]);          // Initialize to equilibrium
    
    // =============================================================================
    // MAIN SIMULATION LOOP
    // =============================================================================
    cout << "Starting simulation..." << endl;

    // Timing
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int t=0; t<maxT; t++) {
        // Step 1: Collision + Streaming (bulk fluid evolution)
        collideStream(distr,distrAdv,&icx[0],&icy[0],&w[0],&stencilOpPt[0],omega);

        // Step 2: Apply boundary conditions (moving lid)
        zouHeBC(distrAdv,uLid);

        // Step 3: Swap arrays for next iteration (ping-pong scheme)
        std::swap(distr,distrAdv);

        // Progress output
        if (t % 1000 == 0) {
            cout << "Completed " << t << " / " << maxT << " time steps" << endl;
        }
    }

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
    writeOutput(distr,icx,icy);
    cout << "Done!" << endl;
    
    return 0;
}   