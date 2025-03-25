#!/bin/bash
nvcc -rdc=true -ccbin clang++ -std=c++11 -I$CUDA_HOME/include -gencode=$NVCC_GENCODE -I $NVSHMEM_HOME/include cavityNVSHMEM.cu -o cavityNVSHMEM -L$NVSHMEM_HOME/lib -lnvshmem -lnvidia-ml -lcuda -lcudart
