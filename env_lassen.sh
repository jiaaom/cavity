#!/bin/bash
module load cuda/12.0.0
export NVSHMEM_HOME=/usr/tce/packages/nvhpc/nvhpc-24.1/Linux_ppc64le/24.1/comm_libs/12.3/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib
export NVCC_GENCODE=arch=compute_70,code=compute_70
export CUDA_HOME=/usr/tce/packages/cuda/cuda-12.0.0
export NVSHMEM_BOOTSTRAP_PLUGIN=$NVSHMEM_HOME/lib/nvshmem_bootstrap_pmix.so
export NVSHMEM_BOOTSTRAP=plugin

