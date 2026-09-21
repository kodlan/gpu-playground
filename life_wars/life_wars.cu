#include <cuda_runtime.h>
#include <stdio.h>
#include "check.h"

int main() {
    int have = 0;

    CUDA_CHECK(cudaGetDeviceCount(&have));
    printf("found %d GPUs\n", have);

    return 0;
}