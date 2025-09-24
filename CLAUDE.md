# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains implementations of a 2D lid-driven cavity flow simulation using the Lattice Boltzmann Method (LBM) with D2Q9 stencil. The project demonstrates performance portability across different GPU programming frameworks and HPC systems, based on the research paper "Performance Evaluation of Heterogeneous GPU Programming Frameworks for Hemodynamic Simulations" by Martin et al. (2023).

## Code Architecture

### Core Implementations
The repository contains multiple implementations of the same cavity flow algorithm, each targeting different programming models and system architectures:

- `cavity.cpp` - Serial CPU implementation (baseline)
- `cavityIntelMPI.cpp` - Intel SYCL + MPI implementation for Aurora/PVC systems
- `cavityCUDAMPI.cu` - CUDA + MPI implementation for NVIDIA systems
- `cavityHIPMPI.hip.cpp` - HIP + MPI implementation for AMD systems
- `cavityHybridMPIISHMEM.cpp` - Hybrid MPI + Intel SHMEM implementation
- `cavityIntelSHMEM.cpp` - Pure Intel SHMEM implementation
- `cavityNVSHEM.cu` - NVIDIA SHMEM implementation
- `cavityRocSHEM.hip.cpp` - ROCm SHMEM implementation

### Algorithm Structure
All implementations follow the same LBM algorithm structure:
- **D2Q9 lattice**: 9-velocity stencil for 2D simulations
- **Grid dimensions**: Configurable via `_LX_` and `_LY_` macros (typically 1024x1024 to 4096x4096)
- **Boundary conditions**: Lid-driven cavity with moving top wall
- **Domain decomposition**: MPI-based domain splitting for parallel execution
- **Halo exchange**: Communication between neighboring MPI ranks for boundary data

### Key Constants and Macros
- `_STENCILSIZE_`: Always 9 for D2Q9 lattice
- `_LX_`, `_LY_`: Grid dimensions in x and y directions
- `_NDIMS_`: Number of dimensions (always 2)
- `_HALO_`: Halo region size for MPI communication
- `_INVALID_`: Marker for invalid grid indices

## Build System

### Local Development with Makefile
For local development and testing, use the included Makefile:

```bash
# Build serial implementation (default)
make cavity

# Build and run serial version
make run

# Build CUDA+MPI version (requires CUDA toolkit)
make cuda-mpi

# Build and run CUDA+MPI on 2 GPUs
make run-cuda-mpi

# Build Intel SYCL+MPI version (requires oneAPI)
make intel-mpi

# Build HIP+MPI version (requires ROCm)
make hip-mpi

# Build Intel SHMEM version
make intel-shmem

# Clean compiled binaries
make clean

# Clean output files
make clean-output

# Show help with all available targets
make help
```

The Makefile handles environment setup and proper compiler flags automatically for each implementation.

### Environment Setup Scripts
Each HPC system has its own environment setup script:
- `env_aurora.sh` - Intel Aurora system (PVC GPUs, oneAPI, Intel SHMEM)
- `env_frontier.sh` - AMD Frontier system (MI250X GPUs, ROCm)
- `env_lassen.sh` - NVIDIA Lassen system (V100 GPUs, CUDA, NVSHMEM)

### Compilation Scripts
Each implementation has a dedicated compilation script:
- `compileIntelMPIAurora.sh` - SYCL implementation for Aurora
- `compileCUDAMPILassen.sh` - CUDA implementation for Lassen
- `compileHIPMPI.sh` - HIP implementation for Frontier
- `compileIntelSHMEMAurora.sh` - Intel SHMEM for Aurora
- `compileNVSHEMLassen.sh` - NVSHMEM for Lassen
- `compileRocSHMEMFrontier.sh` - ROCm SHMEM for Frontier
- `aurora_compile_hybrid.sh` - Hybrid MPI+ISHMEM for Aurora

### Common Build Patterns

#### Intel/Aurora Systems:
```bash
source env_aurora.sh
mpicxx -fsycl -std=c++17 [source].cpp -o [output] -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc"
```

#### NVIDIA/Lassen Systems:
```bash
source env_lassen.sh
mpicxx -std=c++11 -I${CUDA_HOME} [source].cu -o [output] -lcuda -lcudart -L${CUDA_HOME}/lib64
```

#### AMD/Frontier Systems:
```bash
source env_frontier.sh
amdclang++ --std=c++11 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} -x hip --offload-arch=gfx90a [source].hip.cpp -o [output] -lmpi
```

## Running Jobs

### Intel SHMEM Execution
For Intel SHMEM jobs, use the `ishmrun` launcher script which handles device affinity and SYCL device selection automatically. This script:
- Detects available Intel GPUs/tiles using `sycl-ls` or `clinfo`
- Sets appropriate `ZE_AFFINITY_MASK` and `ONEAPI_DEVICE_SELECTOR` environment variables
- Handles both device-level and tile-level GPU assignment
- Uses `numactl` for optimal memory binding

Example usage:
```bash
mpirun -n [num_ranks] ./ishmrun ./cavityIntelSHMEM [args]
```

### Environment Variables
Important environment variables for different systems:
- **Aurora/Intel**: `ZE_FLAT_DEVICE_HIERARCHY`, `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`
- **NVSHMEM**: `NVSHMEM_BOOTSTRAP_PLUGIN`, `NVSHMEM_BOOTSTRAP`, `CUDA_HOME`
- **ROCm**: `ROCM_PATH`, `HIP_ARCH`

## Development Guidelines

### Adding New Implementations
When adding new implementations:
1. Follow the existing naming convention: `cavity[Framework][CommModel].{cpp|cu|hip.cpp}`
2. Maintain the same algorithm structure and constants
3. Create corresponding compilation and environment scripts
4. Ensure proper domain decomposition and halo exchange patterns
5. Use the same output format (`out.txt`) for result verification

### Performance Considerations
- Grid size should be tunable via macros for different system scales
- Memory allocation patterns should match the target architecture (USM, managed memory, etc.)
- Communication patterns are optimized for specific interconnect technologies
- Thread block/work group sizes may need architecture-specific tuning

### Testing and Validation
- All implementations should produce identical results when run with the same parameters
- Serial implementation (`cavity.cpp`) serves as the reference
- Output verification can be done by comparing `out.txt` files across implementations
- MPI implementations create separate output files per rank (e.g., `out_0.txt`, `out_1.txt`)

### Common Commands
- **Build and test locally**: `make run` (builds and runs serial version)
- **Quick CUDA test**: `make run-cuda-mpi` (requires CUDA toolkit and 2 GPUs)
- **Lint/TypeCheck**: No specific commands - this is a research codebase focused on performance portability
- **Clean workspace**: `make clean-all` (removes binaries and output files)