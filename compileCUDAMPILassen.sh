#!/bin/bash
mpicxx -std=c++11 -I/usr/tce/packages/cuda/cuda-11.2.0 cavityCUDAMPI.cu -o cavityCUDAMPI -lcuda -lcudart -L/usr/tce/packages/cuda/cuda-11.2.0/lib64
