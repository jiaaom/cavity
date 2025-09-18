# Makefile for cavity


cudampi:
	nvcc -I/usr/include/openmpi-x86_64/ \
	     -L/usr/lib64/openmpi/lib -lmpi \
		 cavityCUDAMPI.cu -o cudampi
