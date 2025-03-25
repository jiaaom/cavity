#!/bin/bash
mpicxx -fsycl -std=c++17 cavityIntelMPI.cpp -o cavityIntelMPI -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc"