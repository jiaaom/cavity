# Makefile for Lattice Boltzmann Method Cavity Flow Simulations
# Author: Aristotle Martin

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
LDFLAGS = 

# SYCL compiler for Intel implementations (requires oneAPI)
SYCL_CXX = icpx
SYCL_FLAGS = -fsycl -std=c++17 -O3

# CUDA compiler (CUDA 13.0 with correct paths for your system)
CUDA_HOME = /usr/local/cuda-13.0
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_FLAGS = -std=c++11 -O3 -arch=sm_86
CUDA_INCLUDES = -I/usr/include/openmpi-x86_64
CUDA_LIBS = -L/usr/lib64/openmpi/lib -lmpi

# HIP compiler  
HIP_CXX = hipcc
HIP_FLAGS = -std=c++11 -O3

# Source files
SERIAL_SRC = cavity.cpp
INTEL_MPI_SRC = cavityIntelMPI.cpp
CUDA_MPI_SRC = cavityCUDAMPI.cu
HIP_MPI_SRC = cavityHIPMPI.hip.cpp
HYBRID_SRC = cavityHybridMPIISHMEM.cpp
INTEL_SHMEM_SRC = cavityIntelSHMEM.cpp
NVSHMEM_SRC = cavityNVSHEM.cu
ROC_SHMEM_SRC = cavityRocSHEM.hip.cpp

# Executables
SERIAL_EXE = cavity
INTEL_MPI_EXE = cavityIntelMPI
CUDA_MPI_EXE = cavityCUDAMPI
HIP_MPI_EXE = cavityHIPMPI
HYBRID_EXE = cavityHybridMPIISHMEM
INTEL_SHMEM_EXE = cavityIntelSHMEM
NVSHMEM_EXE = cavityNVSHEM
ROC_SHMEM_EXE = cavityRocSHEM

# Default target
.PHONY: all clean help

# Build the serial cavity simulation
cavity: $(SERIAL_SRC)
	@echo "Compiling serial cavity flow simulation..."
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "Successfully built $@"

# Build all available implementations
all: cavity

# Intel SYCL + MPI (requires oneAPI environment)
intel-mpi: $(INTEL_MPI_SRC)
	@echo "Compiling Intel SYCL + MPI implementation..."
	@echo "Note: Requires oneAPI environment (source env_aurora.sh)"
	$(SYCL_CXX) $(SYCL_FLAGS) $< -o $(INTEL_MPI_EXE)
	@echo "Successfully built $(INTEL_MPI_EXE)"

# CUDA + MPI (requires CUDA toolkit)
cuda-mpi: $(CUDA_MPI_SRC)
	@echo "Compiling CUDA + MPI implementation..."
	@echo "Using CUDA 13.0 at $(CUDA_HOME)"
	@echo "Note: Requires CUDA toolkit and MPI"
	export PATH=$(CUDA_HOME)/bin:/usr/lib64/openmpi/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:/usr/lib64/openmpi/lib:$$LD_LIBRARY_PATH && \
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDES) $(CUDA_LIBS) $< -o $(CUDA_MPI_EXE)
	@echo "Successfully built $(CUDA_MPI_EXE)"

# HIP + MPI (requires ROCm)
hip-mpi: $(HIP_MPI_SRC)
	@echo "Compiling HIP + MPI implementation..."
	@echo "Note: Requires ROCm and MPI"
	$(HIP_CXX) $(HIP_FLAGS) $< -o $(HIP_MPI_EXE) -lmpi
	@echo "Successfully built $(HIP_MPI_EXE)"

# Intel SHMEM (requires Intel SHMEM library)
intel-shmem: $(INTEL_SHMEM_SRC)
	@echo "Compiling Intel SHMEM implementation..."
	@echo "Note: Requires Intel SHMEM library (source env_aurora.sh)"
	$(SYCL_CXX) $(SYCL_FLAGS) $< -o $(INTEL_SHMEM_EXE) -lishmem
	@echo "Successfully built $(INTEL_SHMEM_EXE)"

# Run the serial cavity simulation
run: cavity
	@echo "Running cavity flow simulation..."
	./$(SERIAL_EXE)
	@echo "Results written to out.txt"

# Run the CUDA+MPI simulation on 2 GPUs
run-cuda-mpi: cuda-mpi
	@echo "Running CUDA+MPI cavity flow simulation on 2 GPUs..."
	@echo "Grid size: 1024x1024, much larger than serial version!"
	export PATH=$(CUDA_HOME)/bin:/usr/lib64/openmpi/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:/usr/lib64/openmpi/lib:$$LD_LIBRARY_PATH && \
	/usr/lib64/openmpi/bin/mpirun --mca btl tcp,self -np 2 ./$(CUDA_MPI_EXE)
	@echo "Results written to out_0.txt (GPU 0) and out_1.txt (GPU 1)"

# Clean compiled binaries
clean:
	@echo "Cleaning compiled binaries..."
	rm -f $(SERIAL_EXE) $(INTEL_MPI_EXE) $(CUDA_MPI_EXE) $(HIP_MPI_EXE) 
	rm -f $(HYBRID_EXE) $(INTEL_SHMEM_EXE) $(NVSHMEM_EXE) $(ROC_SHMEM_EXE)
	rm -f *.o
	@echo "Clean complete"

# Clean output files
clean-output:
	@echo "Cleaning simulation output files..."
	rm -f out.txt *.dat *.vtk
	@echo "Output files cleaned"

# Clean everything
clean-all: clean clean-output

# Show help information
help:
	@echo "Lattice Boltzmann Method Cavity Flow - Build System"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  cavity        - Build serial cavity flow simulation (default)"
	@echo "  intel-mpi     - Build Intel SYCL + MPI version"
	@echo "  cuda-mpi      - Build CUDA + MPI version"
	@echo "  hip-mpi       - Build HIP + MPI version"
	@echo "  intel-shmem   - Build Intel SHMEM version"
	@echo ""
	@echo "Utility targets:"
	@echo "  run           - Compile and run serial simulation"
	@echo "  run-cuda-mpi  - Compile and run CUDA+MPI on 2 GPUs"
	@echo "  clean         - Remove compiled binaries"
	@echo "  clean-output  - Remove simulation output files"
	@echo "  clean-all     - Remove binaries and output files"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make cavity                    # Build serial version"
	@echo "  make run                       # Build and run serial version"
	@echo "  make cuda-mpi                  # Build CUDA+MPI version"
	@echo "  make run-cuda-mpi              # Build and run CUDA+MPI on 2 GPUs"
	@echo "  make clean                     # Clean compiled files"
	@echo ""
	@echo "Note: Parallel versions require appropriate environments:"
	@echo "  - Intel SYCL: source env_aurora.sh"
	@echo "  - CUDA: source env_lassen.sh"
	@echo "  - HIP: source env_frontier.sh"

# Debug build with symbols and no optimization
debug: CXXFLAGS = -std=c++11 -g -O0 -Wall -Wextra -DDEBUG
debug: $(SERIAL_SRC)
	@echo "Building debug version..."
	$(CXX) $(CXXFLAGS) $< -o $(SERIAL_EXE)_debug
	@echo "Debug build complete: $(SERIAL_EXE)_debug"

# Profile build with profiling information
profile: CXXFLAGS = -std=c++11 -O3 -pg -Wall -Wextra
profile: $(SERIAL_SRC)
	@echo "Building profile version..."
	$(CXX) $(CXXFLAGS) $< -o $(SERIAL_EXE)_profile
	@echo "Profile build complete: $(SERIAL_EXE)_profile"