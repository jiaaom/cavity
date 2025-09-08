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

## Coding Style & Naming Conventions
- Indentation: 4 spaces; no tabs. Keep functions short and single‑purpose.
- C/C++: Prefer concise names; functions use `camelCase` (e.g., `collideStream`), local helpers are short (`myI`, `myJ`).
- Constants/macros: ALL_CAPS with underscores (e.g., `_LX_`, `_STENCILSIZE_`).
- Files: Keep existing names; new variants follow pattern `cavity<Backend><Model>.<ext>`.
- Favor early returns over deep nesting; remove special cases by restructuring rather than branching.

## Testing Guidelines
- No formal test suite yet. Validate changes by:
  - Determinism: run twice and diff outputs (`diff out.txt out.txt.bak`).
  - Regression: compare against `cavity.cpp` on small grids for mass/velocity fields.
  - Multi‑GPU: verify rank boundary continuity between `out_0.txt`/`out_1.txt`.
- Add targeted tests under `tests/` when introduced; name `test_<area>.<ext>` and document expected outputs.

## Commit & Pull Request Guidelines
- Commits: imperative, present tense; short subject, optional scope (e.g., "cuda-mpi: fix halo exchange").
- PRs: include problem statement, approach, perf/correctness impact, validation steps, and environments (GPU/driver/toolkit).
- Link related issues; attach before/after metrics or small diffs for outputs when feasible.

## Security & Configuration Tips
- Avoid hard‑coding cluster paths; use env files and `CUDA_HOME`/`LD_LIBRARY_PATH`.
- Do not commit large outputs; keep them in `.gitignore`.
