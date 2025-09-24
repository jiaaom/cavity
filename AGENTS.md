# Repository Guidelines

## Project Structure & Module Organization
Core solver lives in `cavity.cpp` (serial D2Q9 reference) with GPU backends split across `cavityCUDAMPI.cu`, `cavityCUDA.cu`, `cavityHIPMPI.hip.cpp`, and SHMEM/MPI variants. Helper build and launch scripts reside in the repo root (`Makefile`, `compile*.sh`, `env_*.sh`, `ishmrun`). Generated outputs default to `out.txt` for serial runs and `out_*.txt` for multi-rank GPU runs; keep derived data out of version control. Place new analysis or instrumentation tools under a dedicated subdirectory (e.g., `tools/`) to avoid cluttering the root.

## Build, Test, and Development Commands
- `make cavity` / `make run`: build and execute the serial baseline.
- `make cuda-mpi` / `make run-cuda-mpi`: build and run the CUDA+MPI path (2 ranks by default).
- `make debug`, `make profile`: compile with extra diagnostics or profiling instrumentation.
- `make clean`, `make clean-output`, `make clean-all`: remove build artifacts and generated outputs.
Source the appropriate environment first (`source env_lassen.sh`, `env_frontier.sh`, or `env_aurora.sh`) so compilers and MPI launchers are available.

## Coding Style & Naming Conventions
Use 4-space indentation, K&R braces (`int foo() {`), and descriptive camelCase for functions with ALL_CAPS macros and lattice constants. Favor `std::vector` over raw pointers unless performance dictates otherwise, and keep device kernels grouped by functionality. Mirror existing file suffixes when adding a backend (`.cu`, `.hip.cpp`, etc.) and document assumptions near the kernel launch site.

## Testing Guidelines
No automated test harness is provided; validate changes by running the baseline (`make run`) and comparing `out.txt` against a trusted snapshot, then exercise the relevant accelerator path. Record performance metrics such as MLUPS or runtime deltas when touching numerics. For regression-style checks, stash reference outputs under `tests/data/` (gitignored) and use `diff` to confirm bitwise or tolerance-based agreement.

## Commit & Pull Request Guidelines
Write concise, imperative commit subjects ("Improve CUDA halo exchange"), followed by a brief body explaining motivation and validation. Group unrelated changes into separate commits. Pull requests should summarize algorithmic changes, list tested build targets, and call out performance or precision impacts. Link to design notes or issue IDs where applicable and include profiler traces or timing tables when arguing for hardware-specific optimizations.

## Performance & Acceleration Notes
When proposing FPGA/ASIC pathways, describe pipeline stages (collision, streaming, boundary) and dataflow requirements. Highlight memory access regularity, tile sizes, and precision choices so reviewers can judge feasibility alongside existing GPU implementations.
