# Adaptive Parallel QuickSort

This project implements an Adaptive Parallel QuickSort algorithm using Sequential, OpenMP, and OpenCL processing.

The purpose of the project is to select a suitable sorting method depending on the input workload size.

The final adaptive implementation uses:

- Sequential QuickSort for small workloads
- OpenMP QuickSort for medium workloads
- Hybrid OpenCL + OpenMP for large workloads

The adaptive thresholds were selected based on repeated performance testing.

## Project Structure

```text
project/
│
├── 01_baseline/
│   ├── sequential.cpp
│   ├── openmp.cpp
│   ├── opencl_before_profile.cpp
│   └── quicksort.cl
│
├── 02_opencl_optimization/
│   ├── opencl.cpp
│   └── quicksort.cl
│
├── 03_hybrid/
│   ├── opencl_omp.cpp
│   └── quicksort.cl
│
└── adaptive.cpp
