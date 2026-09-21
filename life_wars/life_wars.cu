#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <string>
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

// Shrink the band by `shrink` in each direction into an RGB image: each output
// pixel shows the colour that has more cells in its shrink x shrink block.
__global__ void downsample(const uint8_t* __restrict__ cur, int rows, int colums, int shrink, uint8_t* rgb, int out_w) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    int out_h = rows / shrink;
    if (ox >= out_w || oy >= out_h)
        return;

    int red = 0, blue = 0;
    for (int dy = 0; dy < shrink; dy++) {
        for (int dx = 0; dx < shrink; dx++) {
            uint8_t v = cur[(oy * shrink + dy + 1) * colums + ox * shrink + dx];
            red += (v == 1);
            blue += (v == 2);
        }
    }

    uint8_t* p = rgb + (oy * out_w + ox) * 3;
    int total = shrink * shrink;

    // brightness = how full the block is; hue = which colour dominates
    int level = (red + blue) * 255 / total;

    if (red > blue) {
        p[0] = 40 + level * 215 / 255;
        p[1] = 20;
        p[2] = 20;
    } else if (blue > red) {
        p[0] = 20;
        p[1] = 60 + level * 100 / 255;
        p[2] = 60 + level * 195 / 255;
    } else if (red) {
        p[0] = p[1] = p[2] = 30 + level * 100 / 255;
    } else {
        p[0] = p[1] = p[2] = 8;
    }
}

static void write_ppm(const std::string& path, const uint8_t* rgb, int w, int h) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        perror(path.c_str()); exit(1);
    }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, (size_t)w * h * 3, f);
    fclose(f);
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

        // the blocking memcpy waits for the kernel, so this is for watching
        // the population only; drop it when timing
        if ((s + 1) % 10 == 0) {
            CUDA_CHECK(cudaMemcpy(grid.data(), cur + W, H * W, cudaMemcpyDeviceToHost));
            int red = 0, blue = 0;
            for (uint8_t v : grid) {
                red += (v == 1);
                blue += (v == 2);
            }
            printf("step %3d: live %6d (red %6d, blue %6d)\n", s + 1, red + blue, red, blue);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    // an out-of-bounds read in a kernel surfaces here, not at the launch
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(grid.data(), cur + W, H * W, cudaMemcpyDeviceToHost));

    int live = 0;
    for (uint8_t v : grid)
        live += (v != 0);
    printf("live cells after %d steps: %d (%.2f%%)\n", steps, live, 100.0 * live / (H * W));

    // render the final grid: shrink x shrink cells per pixel, straight from the
    // device buffer (downsample skips the halo row itself, so pass cur, not cur + W)
    const int shrink = 4;
    const int out_w = W / shrink, out_h = H / shrink;
    size_t rgb_bytes = (size_t)out_w * out_h * 3;

    uint8_t* d_rgb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));

    dim3 og((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    downsample<<<og, block>>>(cur, H, W, shrink, d_rgb, out_w);
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint8_t> rgb(rgb_bytes);
    CUDA_CHECK(cudaMemcpy(rgb.data(), d_rgb, rgb_bytes, cudaMemcpyDeviceToHost));
    write_ppm("frame.ppm", rgb.data(), out_w, out_h);
    printf("wrote frame.ppm (%dx%d)\n", out_w, out_h);

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(cur));
    CUDA_CHECK(cudaFree(nxt));

    return 0;
}