# CUDA Profiling Report

## Run Configuration
- Binary: `./cavityCUDA` built with `nvcc -std=c++11 -O3 -arch=sm_86 -lineinfo`
- Input: 1024×1024 lattice, 100 time steps, Reynolds 100, double precision BGK model
- Tooling: Nsight Compute CLI 2025.3.1 (`ncu`), results captured in `ncu.log`
- Device: Ampere-class GPU (CC 8.6), SM clock ~1.17 GHz, DRAM clock ~7.6 GHz

## Kernel-Level Findings
- `collideStreamKernel` dominates wall time at ~1.23 ms per launch; profiled SM utilization reaches 85–86% of peak issue rate (lines 223–238, 318–320 in `ncu.log`).
- DRAM utilization for the same kernel is only ~21% of peak bandwidth with aggregate memory throughput ~25% of the speed-of-light limit, indicating the kernel is not bandwidth-bound.
- Nsight’s roofline analysis shows 0% of FP32 peak but ~56% of FP64 peak throughput (lines 244–248). The double-precision arithmetic density, not memory pressure, is the limiting factor.
- `zouHeBCKernel` executes in ~6 µs per invocation with sub-2% memory and compute utilization (lines 269–308); its launch grid is too small to affect device-level balance.

## Memory Versus Compute Diagnosis
- For `collideStreamKernel`, SM active cycles (~1.44 M) closely track DRAM active cycles (~1.96 M) but remain compute-heavy: L1/TEX at 35% and L2 at 25% of peak versus SM utilization >85% (lines 253–266, 319–320).
- The kernel’s arithmetic intensity is high because each lattice site performs multiple FP64 exponentiations, dot products, and BGK relaxations for nine directions before a single store. With the achieved bandwidth far below the DRAM roof and no cache-thrashing symptoms, further memory optimizations (e.g., tiling, shared-memory staging) offer limited returns under the current stencil.

## Optimization Opportunities
1. **Precision Strategy** – Evaluate mixed-precision pathways (e.g., FP32 velocity moments with FP64 accumulation) or tuneable fixed-point to reduce reliance on FP64; Nsight projects up to ~2× headroom if FP64 pressure can be relaxed.
2. **Instruction-Level Efficiency** – Revisit the equilibrium computation to collapse redundant multiplies/adds, encourage FMA fusion, and leverage `__dadd_rn`/`__dmul_rn` intrinsics where ordering permits. Any reduction in math instruction count directly increases headroom.
3. **Kernel Fusion Assessment** – Because `zouHeBCKernel` is launch-bound, consider folding lid updates into the main kernel to amortize grid launches if future tuning lowers the main kernel runtime.
4. **Extended Profiling** – Run `ncu --set roofline --kernels ::collideStreamKernel ./cavityCUDA` or gather `sm__sass_thread_inst_executed_op_d*` counters across longer step counts to quantify instruction mix, then align hardware-specific optimizations (e.g., tensor cores for FP32) if physics tolerates.

## Conclusion
The current implementation is compute-bound on FP64 math within `collideStreamKernel`; DRAM bandwidth is underutilized. To unlock additional performance, focus on reducing double-precision pressure and tightening the arithmetic pipeline rather than reworking memory traffic. EOF
