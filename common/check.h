// Shared error-checking and timing helpers
#pragma once
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L  // clock_gettime / CLOCK_MONOTONIC under -std=c11
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__),\
              __FILE__, __LINE__, cudaGetErrorString(err__));                \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#ifdef NCCL_H_
#define NCCL_CHECK(call)                                                     \
  do {                                                                       \
    ncclResult_t res__ = (call);                                             \
    if (res__ != ncclSuccess) {                                              \
      fprintf(stderr, "NCCL error %d at %s:%d: %s\n", (int)res__, __FILE__,  \
              __LINE__, ncclGetErrorString(res__));                          \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)
#endif

#ifdef MPI_VERSION
#define MPI_CHECK(call)                                                      \
  do {                                                                       \
    int rc__ = (call);                                                       \
    if (rc__ != MPI_SUCCESS) {                                               \
      fprintf(stderr, "MPI error %d at %s:%d\n", rc__, __FILE__, __LINE__);  \
      MPI_Abort(MPI_COMM_WORLD, rc__);                                       \
    }                                                                        \
  } while (0)
#endif

// Wall clock in seconds. Use cudaEvent timing for GPU work; use this for
// end-to-end host timing.
static inline double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Convert bytes and seconds to GB/s (decimal gigabytes, like nccl-tests).
static inline double gbps(double bytes, double seconds) {
  return bytes / seconds / 1e9;
}