#!/bin/bash
source ./env_frontier.sh
amdclang++ --std=c++11 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} -x hip --offload-arch=gfx90a -I/opt/cray/pe/mpich/8.1.31/ofi/crayclang/17.0/include cavityHIPMPI.hip.cpp -o cavityHIPMPI -lmpi -L/opt/cray/pe/mpich/8.1.31/ofi/crayclang/17.0/lib