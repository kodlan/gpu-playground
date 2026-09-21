# Shared make settings.
#
# Override on the command line, for example:
#   make CUDA_HOME=/usr/local/cuda-13.0 NCCL_HOME=$HOME/opt/nccl
#
# GENCODE lists two GPUs: sm_75 (RTX 2070) and sm_120 (RTX 5070 Ti).

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
MPI_HOME  ?= /usr/lib/x86_64-linux-gnu/openmpi
NVCC      ?= $(CUDA_HOME)/bin/nvcc
MPICC     ?= mpicc
MPICXX    ?= mpicxx

GENCODE ?= -gencode arch=compute_75,code=sm_75 -gencode arch=compute_120,code=sm_120

COMMON_DIR := $(dir $(lastword $(MAKEFILE_LIST)))common
CXXFLAGS   += -O2 -std=c++17 -I$(COMMON_DIR)
NVCCFLAGS  += -O2 -std=c++17 -lineinfo $(GENCODE) -I$(COMMON_DIR)
NCCL_INC   := -I$(NCCL_HOME)/include
NCCL_LIB   := -L$(NCCL_HOME)/lib -Xlinker -rpath=$(NCCL_HOME)/lib -lnccl
MPI_INC    := -I$(MPI_HOME)/include -DOMPI_SKIP_MPICXX
MPI_LIB    := -L$(MPI_HOME)/lib -lmpi

.PHONY: all clean run