#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <utility>
#include <vector>
#include "../common/check.h"


__global__ void life_step(const uint8_t* cur, uint8_t* next, int rows, int columns) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= columns || r >= rows)
         return; // threads past the edge do nothing

    int y = r + 1; // +1 skips the top halo row

    int left = (c == 0) ? columns - 1 : c - 1;
    int right = (c == columns - 1) ? 0 : c + 1;

    int alive = 0, red = 0;

    #define LOOK(yy, xx) { \
        uint8_t v = cur[(yy) * columns + (xx)];    \
        alive += (v != 0);                     \
        red += (v == 1);                       \
    }

    LOOK(y - 1, left);
    LOOK(y - 1, c);
    LOOK(y - 1, right);
    LOOK(y, left);
    LOOK(y, right);
    LOOK(y + 1, left);
    LOOK(y + 1, c);
    LOOK(y + 1, right);

    #undef LOOK

    uint8_t me = cur[y * columns + c];
    uint8_t out = 0;

    if (me != 0)
        out = (alive == 2 || alive == 3) ? me : 0;
    else if (alive == 3)
        out = (red >= 2) ? 1 : 2;

    next[y * columns + c] = out;
}

int main() {
    int have = 0;

    CUDA_CHECK(cudaGetDeviceCount(&have));
    printf("found %d GPUs\n", have);

    const int W = 512, H = 512;
    const int steps = 100;

    // random soup: ~30% alive, colour 1 in the top half, 2 in the bottom half
    std::vector<uint8_t> grid(H * W);
    int initial = 0;
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            bool live = rand() % 100 < 30;
            grid[r * W + c] = live ? (r < H / 2 ? 1 : 2) : 0;
            initial += live;
        }
    }
    printf("initial live cells: %d of %d\n", initial, H * W);

    // device buffers carry one halo row above and below; zeroing keeps them dead
    size_t bytes = (size_t)(H + 2) * W;
    uint8_t *cur = nullptr, *nxt = nullptr;
    CUDA_CHECK(cudaMalloc(&cur, bytes));
    CUDA_CHECK(cudaMalloc(&nxt, bytes));
    CUDA_CHECK(cudaMemset(cur, 0, bytes));
    CUDA_CHECK(cudaMemset(nxt, 0, bytes));

    // + W skips the top halo row
    CUDA_CHECK(cudaMemcpy(cur + W, grid.data(), H * W, cudaMemcpyHostToDevice));

    // g.x * g.y blocks of 32 * 8 threads each: enough to cover every cell,
    // the bounds guard in the kernel eats the extra threads
    dim3 block(32, 8);
    dim3 g((W + 31) / 32, (H + 7) / 8);

    for (int s = 0; s < steps; s++) {
        life_step<<<g, block>>>(cur, nxt, H, W);
        std::swap(cur, nxt);
    }
    CUDA_CHECK(cudaGetLastError());
    // an out-of-bounds read in a kernel surfaces here, not at the launch
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(grid.data(), cur + W, H * W, cudaMemcpyDeviceToHost));

    int live = 0;
    for (uint8_t v : grid)
        live += (v != 0);
    printf("live cells after %d steps: %d (%.2f%%)\n", steps, live, 100.0 * live / (H * W));

    CUDA_CHECK(cudaFree(cur));
    CUDA_CHECK(cudaFree(nxt));

    return 0;
}