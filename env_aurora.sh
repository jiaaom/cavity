module load cmake/3.27.9
module load oneapi/release/2024.2.1
export INTEL_SHMEM_INSTALLPATH="/lus/flare/projects/TwinBlood/shared_executables/intelshmem/intel_shmem_install_oneapi2024.2.1"
export PATH=${INTEL_SHMEM_INSTALLPATH}/bin:${PATH}
export C_INCLUDE_PATH=${INTEL_SHMEM_INSTALLPATH}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${INTEL_SHMEM_INSTALLPATH}/include:${CPLUS_INCLUDE_PATH}
export LD_LIBRARY_PATH=${INTEL_SHMEM_INSTALLPATH}/lib:${LD_LIBRARY_PATH}
export ISHMEM_RUNTIME="MPI"