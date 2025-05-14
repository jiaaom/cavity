export ISH_INSTALL="/lus/flare/projects/TwinBlood/wdl/scaling_study/shmem/prints/ishmem_install"
export PATH=${ISH_INSTALL}/bin:${PATH}
export C_INCLUDE_PATH=${ISH_INSTALL}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${ISH_INSTALL}/include:${CPLUS_INCLUDE_PATH}
export LD_LIBRARY_PATH=${ISH_INSTALL}/lib:${LD_LIBRARY_PATH}
export LD_RUN_PATH=${ISH_INSTALL}/lib:${LD_RUN_PATH}

export ISHMEM_SYMMETRIC_SIZE=10000000000
export ISHMEM_RUNTIME=MPI
