#!/bin/bash
module reset
module load rocm/6.2.4
module load craype-network-ucx
module load cray-mpich-ucx/8.1.31
module load cray-ucx/2.7.0-1
export ROCSHMEM_ROOT=/lustre/orion/med113/proj-shared/amartin/pgas/rocSHMEM
export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_ROOT}/build/install
export MPICH_UCX_INSTALL_DIR=/opt/cray/pe/mpich/8.1.31/ucx/crayclang/17.0
export PATH=${ROCSHMEM_INSTALL_DIR}/bin:${PATH}
export CPLUS_INCLUDE_PATH=${ROCSHMEM_INSTALL_DIR}/include:${CPLUS_INCLUDE_PATH}
export C_INCLUDE_PATH=${ROCSHMEM_INSTALL_DIR}/include:${C_INCLUDE_PATH}
export LD_LIBRARY_PATH=${ROCSHMEM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

hipcc -c -fgpu-rdc -x hip cavityRocSHMEM.hip.cpp \
  -I/opt/rocm/include \
  -I$ROCSHMEM_INSTALL_DIR/include \
  -I$MPICH_UCX_INSTALL_DIR/include/

hipcc -fgpu-rdc --hip-link cavityRocSHMEM.hip.o -o cavityRocSHMEM \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
  $MPICH_UCX_INSTALL_DIR/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64