#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <vector>
#include <omp.h> // OpenMP header

#define _STENCILSIZE_ 9
#define _LX_ 128
#define _LY_ 128
#define _NDIMS_ 2
#define _INVALID_ -1

using namespace std;

int getGridIdx(int i, int j) {
    if (i < 0 || i >= _LX_ || j < 0 || j >= _LY_) return _INVALID_;
    return i + _LX_ * j;
}

void writeOutput(vector<double>& distr, int* icx, int* icy, int numThreads) {
    std::ofstream out_file("out.txt");

    #pragma omp parallel for num_threads(numThreads) collapse(2)
    for (int idxI = 0; idxI < _LX_; idxI++) {
        for (int idxJ = 0; idxJ < _LY_; idxJ++) {
            int idxIJ = getGridIdx(idxI, idxJ);
            double rho = 0.0, ux = 0.0, uy = 0.0;
            double distr_local[_STENCILSIZE_];

            for (int iNbr = 0; iNbr < _STENCILSIZE_; iNbr++) {
                distr_local[iNbr] = distr[idxIJ*_STENCILSIZE_+iNbr];
                rho += distr_local[iNbr];
                ux += distr_local[iNbr] * icx[iNbr];
                uy += distr_local[iNbr] * icy[iNbr];
            }

            double orho = 1.0 / rho;
            ux *= orho;
            uy *= orho;

            #pragma omp critical
            out_file << std::setprecision(16) << idxI << ", " << idxJ << ", "
                     << ux << ", " << uy << ", " << rho << std::endl;
        }
    }
    out_file.close();
}

void zouHeBC(vector<double>& distr, double uLid) {
    int myJ = 0;
    for (int myI = 0; myI < _LX_; myI++) {
        int idxIJ = getGridIdx(myI, myJ);
        double ux = uLid, uy = 0.0;
        double rho = (1.0/(1.0-uy))*(distr[idxIJ*_STENCILSIZE_+0]+
                                     distr[idxIJ*_STENCILSIZE_+1]+
                                     distr[idxIJ*_STENCILSIZE_+3]+
                                     2*(distr[idxIJ*_STENCILSIZE_+4]+
                                        distr[idxIJ*_STENCILSIZE_+7]+
                                        distr[idxIJ*_STENCILSIZE_+8]));
        distr[idxIJ*_STENCILSIZE_+2] = distr[idxIJ*_STENCILSIZE_+4] + (2.0/3.0)*rho*uy;
        distr[idxIJ*_STENCILSIZE_+5] = distr[idxIJ*_STENCILSIZE_+7] - 
                                       (1.0/2.0)*(distr[idxIJ*_STENCILSIZE_+1] - distr[idxIJ*_STENCILSIZE_+3]) + 
                                       (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
        distr[idxIJ*_STENCILSIZE_+6] = distr[idxIJ*_STENCILSIZE_+8] + 
                                       (1.0/2.0)*(distr[idxIJ*_STENCILSIZE_+1] - distr[idxIJ*_STENCILSIZE_+3]) - 
                                       (1.0/2.0)*rho*ux - (1.0/6.0)*rho*uy;
    }
}

void collideStream(vector<double>& distr, vector<double>& distrAdv, int* icx, int* icy, double* w, int* stencilOpPt, double omega, int numThreads) {
    #pragma omp parallel for num_threads(numThreads)
    for (int ii=0; ii<_LX_*_LY_; ii++) {
        double rho = 0.0, ux = 0.0, uy = 0.0;
        double distr_local[_STENCILSIZE_];

        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            distr_local[iNbr] = distr[ii*_STENCILSIZE_+iNbr];
            rho += distr_local[iNbr];
            ux += distr_local[iNbr]*icx[iNbr];
            uy += distr_local[iNbr]*icy[iNbr];
        }

        double orho = 1.0 / rho;
        ux *= orho;
        uy *= orho;
        double uke = ux*ux + uy*uy;

        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            int nbrInd = stencilOpPt[ii*_STENCILSIZE_+iNbr];
            double cdotu = icx[iNbr]*ux + icy[iNbr]*uy;
            double distr_eq = w[iNbr] * rho * (1.0 + 3.0*cdotu + 4.5*cdotu*cdotu - 1.5*uke);
            distrAdv[nbrInd] = omega*distr_eq + (1.0-omega)*distr_local[iNbr];
        }
    }
}

void setupAdjacency(vector<int>& stencilOpPt, int* icx, int* icy, int* opp, int numThreads) {
    #pragma omp parallel for num_threads(numThreads)
    for (int ii=0; ii<_LX_*_LY_; ii++) {
        int myI = ii % _LX_;
        int myJ = ii / _LX_;

        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            int nbrI = myI + icx[iNbr];
            int nbrJ = myJ + icy[iNbr];
            int nbrIJ = getGridIdx(nbrI, nbrJ);
            if (nbrIJ < 0) {
                stencilOpPt[ii*_STENCILSIZE_+iNbr] = ii*_STENCILSIZE_ + opp[iNbr];
            } else {
                stencilOpPt[ii*_STENCILSIZE_+iNbr] = nbrIJ*_STENCILSIZE_ + iNbr;
            }
        }
    }
}

void initializeFluid(vector<double>& distr, vector<double>& distrAdv, double* w, int numThreads) {
    #pragma omp parallel for num_threads(numThreads)
    for (int ii=0; ii<_LX_*_LY_; ii++) {
        for (int iNbr=0; iNbr<_STENCILSIZE_; iNbr++) {
            distr[ii*_STENCILSIZE_+iNbr] = w[iNbr];
            distrAdv[ii*_STENCILSIZE_+iNbr] = 0.0;
        }
    }
}

void setupGrid(vector<int>& fluidPts, int numThreads) {
    #pragma omp parallel for num_threads(numThreads) collapse(2)
    for (int idxI=0; idxI<_LX_; idxI++) {
        for (int idxJ=0; idxJ<_LY_; idxJ++) {
            int idxIJ = idxI + _LX_ * idxJ;
            fluidPts[idxIJ*_NDIMS_] = idxI;
            fluidPts[idxIJ*_NDIMS_+1] = idxJ;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <numThreads>" << endl;
        return 1;
    }
    int numThreads = atoi(argv[1]);
    cout << "Running with " << numThreads << " threads." << endl;

    int maxT = 10000;
    double uLid = 0.05, Re = 100.0;
    double cs2 = 1.0/3.0;
    double nu = uLid * _LX_ / Re;
    double omega = 1.0 / (3.0*nu + 0.5);

    int icx[_STENCILSIZE_] = {0,1,0,-1,0,1,-1,-1,1};
    int icy[_STENCILSIZE_] = {0,0,1,0,-1,1,1,-1,-1};
    int opp[_STENCILSIZE_] = {0,3,4,1,2,7,8,5,6};
    double w[_STENCILSIZE_] = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,
                               1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};

    vector<double> distr(_LX_*_LY_*_STENCILSIZE_);
    vector<double> distrAdv(_LX_*_LY_*_STENCILSIZE_);
    vector<int> stencilOpPt(_LX_*_LY_*_STENCILSIZE_);
    vector<int> fluidPts(_LX_*_LY_*_NDIMS_);

    cout << "Initializing 2D Lid-Driven Cavity Flow..." << endl;
    setupGrid(fluidPts,numThreads);
    setupAdjacency(stencilOpPt,icx,icy,opp,numThreads);
    initializeFluid(distr,distrAdv,w,numThreads);

    cout << "Starting simulation..." << endl;
    for (int t=0; t<maxT; t++) {
        collideStream(distr,distrAdv,icx,icy,w,&stencilOpPt[0],omega,numThreads);
        zouHeBC(distrAdv,uLid);
        swap(distr,distrAdv);
        if (t % 1000 == 0) cout << "Completed " << t << " / " << maxT << " time steps" << endl;
    }

    cout << "Writing results to out.txt..." << endl;
    writeOutput(distr,icx,icy,numThreads);
    cout << "Done!" << endl;
    return 0;
}
