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
CUDA_HOME = /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_FLAGS = -std=c++11 -O3 -arch=sm_86

# Detect machine architecture
ARCH := $(shell uname -m)

# Set OpenMPI path based on architecture
ifeq ($(ARCH), x86_64)
    OPENMPI_PATH = /usr/lib64/openmpi
else ifeq ($(ARCH), aarch64)
    OPENMPI_PATH = /usr/lib/aarch64-linux-gnu/openmpi
else
    $(warning "Unsupported architecture $(ARCH). OpenMPI paths may be incorrect.")
    OPENMPI_PATH = /usr/lib64/openmpi
endif

# Resolve MPI include path; prefer user override via OPENMPI_INCLUDE if set
OPENMPI_INCLUDE ?=
OPENMPI_INCLUDE_SEARCH := $(if $(OPENMPI_INCLUDE),$(OPENMPI_INCLUDE) ,) \
                           $(OPENMPI_PATH)/include \
                           /usr/include/openmpi-$(ARCH) \
                           /usr/include/openmpi_$(ARCH) \
                           /usr/include/openmpi-$(shell uname -m) \
                           /usr/include/openmpi-x86_64
OPENMPI_INCLUDE_DIR := $(firstword $(foreach dir,$(OPENMPI_INCLUDE_SEARCH),$(if $(wildcard $(dir)/mpi.h),$(dir))))
ifeq ($(strip $(OPENMPI_INCLUDE_DIR)),)
    $(warning "Could not automatically locate mpi.h; defaulting to $(OPENMPI_PATH)/include")
    OPENMPI_INCLUDE_DIR := $(OPENMPI_PATH)/include
endif

# Resolve MPI library path; allow user override via OPENMPI_LIB if set
OPENMPI_LIB ?=
OPENMPI_LIB_SEARCH := $(if $(OPENMPI_LIB),$(OPENMPI_LIB) ,) \
                       $(OPENMPI_PATH)/lib \
                       $(OPENMPI_PATH)/lib64 \
                       /usr/lib64/openmpi/lib \
                       /usr/lib64 \
                       /usr/lib
OPENMPI_LIB_DIR := $(firstword $(foreach dir,$(OPENMPI_LIB_SEARCH),$(if $(wildcard $(dir)/libmpi.*),$(dir))))
ifeq ($(strip $(OPENMPI_LIB_DIR)),)
    $(warning "Could not automatically locate libmpi.*; defaulting to $(OPENMPI_PATH)/lib")
    OPENMPI_LIB_DIR := $(OPENMPI_PATH)/lib
endif

CUDA_INCLUDES = -I$(OPENMPI_INCLUDE_DIR)
CUDA_LIBS = -L$(OPENMPI_LIB_DIR) -lmpi

# HIP compiler  
HIP_CXX = hipcc
HIP_FLAGS = -std=c++11 -O3

# Source files
SERIAL_SRC = cavity.cpp
CUDA_SRC = cavityCUDA.cu
INTEL_MPI_SRC = cavityIntelMPI.cpp
CUDA_MPI_SRC = cavityCUDAMPI.cu
HIP_MPI_SRC = cavityHIPMPI.hip.cpp
HYBRID_SRC = cavityHybridMPIISHMEM.cpp
INTEL_SHMEM_SRC = cavityIntelSHMEM.cpp
NVSHMEM_SRC = cavityNVSHEM.cu
ROC_SHMEM_SRC = cavityRocSHEM.hip.cpp

# Executables
SERIAL_EXE = cavity
CUDA_EXE = cavityCUDA
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

# Single GPU CUDA (requires CUDA toolkit)
cuda: $(CUDA_SRC)
	@echo "Compiling single GPU CUDA implementation..."
	@echo "Using CUDA 13.0 at $(CUDA_HOME)"
	@echo "Note: Requires CUDA toolkit (no MPI needed)"
	export PATH=$(CUDA_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH && \
	$(NVCC) $(CUDA_FLAGS) $< -o $(CUDA_EXE)
	@echo "Successfully built $(CUDA_EXE)"

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
	export PATH=$(CUDA_HOME)/bin:$(OPENMPI_PATH)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$(OPENMPI_PATH)/lib:$$LD_LIBRARY_PATH && \
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

# NVSHMEM (requires NVSHMEM library)
nvshmem: $(NVSHMEM_SRC)
	@echo "Compiling NVSHMEM implementation..."
	@echo "Note: Requires NVSHMEM library and CUDA toolkit"
	export PATH=$(CUDA_HOME)/bin:$(OPENMPI_PATH)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$(OPENMPI_PATH)/lib:$$LD_LIBRARY_PATH && \
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDES) $< -o $(NVSHMEM_EXE) -lnvshmem -lcuda -lcudart
	@echo "Successfully built $(NVSHMEM_EXE)"

# Run the serial cavity simulation
run: cavity
	@echo "Running cavity flow simulation..."
	./$(SERIAL_EXE)
	@echo "Results written to out.txt"


# OpenMP multithreaded LBM cavity flow
OMP_SRC = cavity_omp.cpp
OMP_EXE = cavity_omp

# Build OpenMP version
run-omp: cavity_omp.cpp
	@echo "Compiling OpenMP multithreaded LBM cavity flow simulation..."
	$(CXX) $(CXXFLAGS) -fopenmp $< -o cavity_omp
	@echo "Successfully built cavity_omp"
	@echo "Running OpenMP simulation..."
	@if [ -z "$(NTHREADS)" ]; then \
	    echo "No NTHREADS specified, defaulting to 4 threads"; \
	    ./cavity_omp 4; \
	else \
	    ./cavity_omp $(NTHREADS); \
	fi
# Run the single GPU CUDA simulation
run-cuda: cuda
	@echo "Running single GPU CUDA cavity flow simulation..."
	@echo "Grid size: 1024x1024, optimized for GPU execution!"
	export PATH=$(CUDA_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH && \
	./$(CUDA_EXE)
	@echo "Results written to out.txt"

# Run the CUDA+MPI simulation on 2 GPUs
run-cuda-mpi: cuda-mpi
	@echo "Running CUDA+MPI cavity flow simulation on 2 GPUs..."
	@echo "Grid size: 1024x1024, much larger than serial version!"
	export PATH=$(CUDA_HOME)/bin:$(OPENMPI_PATH)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$(OPENMPI_PATH)/lib:$$LD_LIBRARY_PATH && \
	mpirun --mca btl tcp,self -np 2 ./$(CUDA_MPI_EXE)
	@echo "Results written to out_0.txt (GPU 0) and out_1.txt (GPU 1)"

# Run the NVSHMEM simulation on 2 GPUs
run-nvshmem: nvshmem
	@echo "Running NVSHMEM cavity flow simulation on 2 GPUs..."
	@echo "Grid size: 1024x1024, SHMEM-based communication"
	export PATH=$(CUDA_HOME)/bin:$(OPENMPI_PATH)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$(OPENMPI_PATH)/lib:$$LD_LIBRARY_PATH && \
	nvshmrun -np 2 ./$(NVSHMEM_EXE)
	@echo "Results written to out_0.txt (GPU 0) and out_1.txt (GPU 1)"

# Clean compiled binaries
clean:
	@echo "Cleaning compiled binaries..."
	rm -f $(SERIAL_EXE) $(CUDA_EXE) $(INTEL_MPI_EXE) $(CUDA_MPI_EXE) $(HIP_MPI_EXE)
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
	@echo "  cuda          - Build single GPU CUDA version"
	@echo "  intel-mpi     - Build Intel SYCL + MPI version"
	@echo "  cuda-mpi      - Build CUDA + MPI version"
	@echo "  hip-mpi       - Build HIP + MPI version"
	@echo "  intel-shmem   - Build Intel SHMEM version"
	@echo "  nvshmem       - Build NVSHMEM version"
	@echo ""
	@echo "Utility targets:"
	@echo "  run           - Compile and run serial simulation"
	@echo "  run-cuda      - Compile and run single GPU CUDA simulation"
	@echo "  run-cuda-mpi  - Compile and run CUDA+MPI on 2 GPUs"
	@echo "  run-nvshmem   - Compile and run NVSHMEM on 2 GPUs"
	@echo "  clean         - Remove compiled binaries"
	@echo "  clean-output  - Remove simulation output files"
	@echo "  clean-all     - Remove binaries and output files"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make cavity                    # Build serial version"
	@echo "  make run                       # Build and run serial version"
	@echo "  make cuda                      # Build single GPU CUDA version"
	@echo "  make run-cuda                  # Build and run single GPU CUDA"
	@echo "  make cuda-mpi                  # Build CUDA+MPI version"
	@echo "  make run-cuda-mpi              # Build and run CUDA+MPI on 2 GPUs"
	@echo "  make nvshmem                   # Build NVSHMEM version"
	@echo "  make run-nvshmem               # Build and run NVSHMEM on 2 GPUs"
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
