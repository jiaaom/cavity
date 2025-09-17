# Repository Guidelines

## Project Structure & Module Organization
- `cavity.cpp`: Serial D2Q9 LBM cavity solver (reference).
- `cavityCUDAMPI.cu`: CUDA + MPI implementation (multi‑GPU).
- `cavityHIPMPI.hip.cpp`, `cavityIntelMPI.cpp`, `cavityIntelSHMEM.cpp`, `cavityHybridMPIISHMEM.cpp`, `cavityNVSHEM.cu`, `cavityRocSHEM.hip.cpp`: Alternate backends.
- `Makefile`: Unified build and run targets.
- `env_*.sh`, `compile*.sh`, `ishmrun`: Environment helpers and cluster scripts.
- Outputs: `out.txt` (serial), `out_0.txt`, `out_1.txt` (CUDA+MPI).

## Build, Test, and Development Commands
- Build serial: `make cavity`
- Run serial: `make run`
- Build CUDA+MPI: `make cuda-mpi` (requires CUDA + MPI)
- Run CUDA+MPI (2 ranks): `make run-cuda-mpi`
- Debug/Profiling: `make debug`, `make profile`
- Clean: `make clean`, `make clean-output`, `make clean-all`
Notes: Source the right env first (`source env_lassen.sh`, `env_frontier.sh`, or `env_aurora.sh`). GPU runs may require `mpirun` availability and device visibility.

# Goal

My goal is to explore the possibility for hardware acceleration (by FPGA or ASIC) for LBM algorithm used here.
Therefore, I'm generally interested in analyzing the following aspects:

- Parallelism Opportunities: Examine whether the algorithm has inherent parallelism - data parallelism (same operation on multiple data), task parallelism (different operations running concurrently), or pipeline parallelism (streaming data through sequential stages).
- Data Dependencies: Analyze the dependency graph of your algorithm. Algorithms with minimal data dependencies between operations are easier to parallelize. Look for loop-carried dependencies that might limit parallelization opportunities.

- Memory Access Patterns: Regular, predictable access patterns (like sequential or strided access) work well in hardware. Random access patterns can be problematic due to memory hierarchy limitations.
- Data Locality: Algorithms that reuse data frequently (temporal locality) or access nearby memory locations (spatial locality) are more suitable for hardware acceleration with local memory hierarchies.
- Memory Bandwidth Requirements: Calculate whether your algorithm is compute-bound or memory-bound. Memory-bound algorithms may not benefit significantly from acceleration unless you can improve memory bandwidth utilization.
- Streaming vs. Random Access: Streaming algorithms that process data in a pipeline fashion are often ideal for hardware implementation.

- Precision Flexibility: Can your algorithm tolerate reduced precision (fixed-point vs. floating-point)? Hardware implementations often benefit from custom precision arithmetic.
- Numerical Stability: Ensure the algorithm remains numerically stable with potential precision reductions or different arithmetic implementations.
