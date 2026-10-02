#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "../common/check.h"

static const int W = 2048, H = 2048;


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

// Run `steps` generations of `init` split into n horizontal bands, band r on
// GPU r, and return the final grid. Band r is a (rows[r] + 2) x W buffer with
// one halo row above and one below. The outer halos (above band 0, below the
// last band) stay dead; the inner ones, between band 0 and band 1, are
// refreshed from the neighbour's edge row before every step with a peer copy.
// With n == 1 there is one band on GPU 0 and no exchange: the single-buffer
// program from before.
//
// Every CUDA call for band r is made with GPU r current (cudaSetDevice), and
// band r's kernels go to streams[r]. One stream per GPU is not for overlap
// (each device has a default stream already): NCCL calls take a stream, and
// the kernel and the NCCL work for one GPU belong in the same queue so they
// run in order without the host waiting in between.
//
// `verbose` prints the live count every 10 steps; if `rgb` is given, the
// final grid is rendered into it at 4 x 4 cells per pixel.
static std::vector<uint8_t> simulate(const std::vector<uint8_t>& init, int n, int steps, bool verbose, std::vector<uint8_t>* rgb) {
    // band r owns grid rows [row0[r], row0[r] + rows[r])
    int rows[2] = {0, 0}, row0[2] = {0, 0};
    for (int r = 0; r < n; r++) {
        rows[r] = H / n + (r < H % n ? 1 : 0);
        row0[r] = (r == 0) ? 0 : row0[r - 1] + rows[r - 1];
    }

    // two buffers per band on its own GPU: the kernel reads cur and writes nxt,
    // then they swap. Zeroing keeps the halo rows dead until something is
    // copied into them.
    uint8_t *cur[2] = {nullptr, nullptr}, *nxt[2] = {nullptr, nullptr};
    cudaStream_t streams[2] = {nullptr, nullptr};
    for (int r = 0; r < n; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));

        size_t bytes = (size_t)(rows[r] + 2) * W;
        CUDA_CHECK(cudaMalloc(&cur[r], bytes));
        CUDA_CHECK(cudaMalloc(&nxt[r], bytes));
        CUDA_CHECK(cudaMemset(cur[r], 0, bytes));
        CUDA_CHECK(cudaMemset(nxt[r], 0, bytes));
        // + W skips the top halo row
        CUDA_CHECK(cudaMemcpy(cur[r] + W, init.data() + (size_t)row0[r] * W,
                              (size_t)rows[r] * W, cudaMemcpyHostToDevice));
    }

    // copy every band's owned rows (not the halos) back into one host grid
    std::vector<uint8_t> grid(init.size());
    auto gather = [&]() {
        for (int r = 0; r < n; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(grid.data() + (size_t)row0[r] * W, cur[r] + W,
                                  (size_t)rows[r] * W, cudaMemcpyDeviceToHost));
        }
    };

    // blocks of 32 * 8 threads, enough to cover every cell of the band;
    // the bounds guard in the kernel eats the extra threads
    dim3 block(32, 8);

    for (int s = 0; s < steps; s++) {
        // Halo exchange BEFORE the step: the kernel reads the halo rows, so
        // they must hold the neighbour's current edge row when it runs. The
        // first step would otherwise see empty halos and the bands drift apart.
        //
        // cudaMemcpyPeer(dst, dst device, src, src device, bytes) copies
        // between GPUs; without peer-to-peer access the driver stages it
        // through host memory. It is ordered after all pending work on both
        // devices and before everything queued after it, so the previous
        // step's kernels are done when it reads, and this step's wait for it.
        if (n > 1) {
            // band 0's last owned row (GPU 0) -> band 1's top halo row (GPU 1, row 0 of its buffer)
            CUDA_CHECK(cudaMemcpyPeer(cur[1], 1, cur[0] + (size_t)rows[0] * W, 0, W));
            // band 1's first owned row (GPU 1) -> band 0's bottom halo row (GPU 0, row rows[0] + 1)
            CUDA_CHECK(cudaMemcpyPeer(cur[0] + (size_t)(rows[0] + 1) * W, 0, cur[1] + W, 1, W));
        }

        for (int r = 0; r < n; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            dim3 g((W + block.x - 1) / block.x, (rows[r] + block.y - 1) / block.y);
            life_step<<<g, block, 0, streams[r]>>>(cur[r], nxt[r], rows[r], W);
            std::swap(cur[r], nxt[r]);
        }

        // the blocking memcpy waits for the kernels, so this is for watching
        // the population only; drop it when timing
        if (verbose && (s + 1) % 10 == 0) {
            gather();
            int red = 0, blue = 0;
            for (uint8_t v : grid) {
                red += (v == 1);
                blue += (v == 2);
            }
            printf("step %3d: live %6d (red %6d, blue %6d)\n", s + 1, red + blue, red, blue);
        }
    }
    for (int r = 0; r < n; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaGetLastError());
        // an out-of-bounds read in a kernel surfaces here, not at the launch
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    gather();

    if (rgb) {
        // render each band on its own GPU, straight from the device buffer
        // (downsample skips the halo row itself, so pass cur, not cur + W),
        // then copy each band's strip into its slice of the host image. Band r
        // covers output rows from row0[r] / shrink; rows[r] is a multiple of
        // shrink for H = 2048 and n <= 2.
        const int shrink = 4;
        const int out_w = W / shrink, out_h = H / shrink;
        rgb->resize((size_t)out_w * out_h * 3);

        for (int r = 0; r < n; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            int band_h = rows[r] / shrink;
            size_t strip_bytes = (size_t)out_w * band_h * 3;

            uint8_t* d_rgb = nullptr;
            CUDA_CHECK(cudaMalloc(&d_rgb, strip_bytes));

            dim3 og((out_w + block.x - 1) / block.x, (band_h + block.y - 1) / block.y);
            downsample<<<og, block, 0, streams[r]>>>(cur[r], rows[r], W, shrink, d_rgb, out_w);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(rgb->data() + (size_t)(row0[r] / shrink) * out_w * 3, d_rgb,
                                  strip_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_rgb));
        }
    }

    for (int r = 0; r < n; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaFree(cur[r]));
        CUDA_CHECK(cudaFree(nxt[r]));
        CUDA_CHECK(cudaStreamDestroy(streams[r]));
    }
    return grid;
}

static void usage(const char* prog) {
    fprintf(stderr,
            "usage: %s [--gpus N] [--steps N] [--seed N] [--check]\n"
            "  --gpus N   number of GPUs, 1 or 2; the grid is split into one band per GPU\n"
            "  --steps N  generations to run (default 100)\n"
            "  --seed N   seed for the random soup (default 1)\n"
            "  --check    with --gpus 2, also run one band and compare the grids\n",
            prog);
    exit(2);
}

int main(int argc, char** argv) {
    int n = 1;          // number of GPUs, one band each
    int steps = 100;
    unsigned seed = 1;
    bool check = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpus" && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc)
            steps = atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc)
            seed = (unsigned)strtoul(argv[++i], nullptr, 10);
        else if (a == "--check")
            check = true;
        else
            usage(argv[0]);
    }
    if (n < 1 || n > 2) {
        fprintf(stderr, "--gpus must be 1 or 2\n");
        return 2;
    }

    int have = 0;
    CUDA_CHECK(cudaGetDeviceCount(&have));
    if (have < n) {
        fprintf(stderr, "--gpus %d but only %d GPU%s found\n", n, have, have == 1 ? "" : "s");
        return 2;
    }
    printf("found %d GPUs, using %d\n", have, n);

    // random soup: ~30% alive, colour 1 in the top half, 2 in the bottom half.
    // Built once, so every simulate() call below starts from the same grid.
    srand(seed);
    std::vector<uint8_t> soup(H * W);
    int initial = 0;
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            bool live = rand() % 100 < 30;
            soup[r * W + c] = live ? (r < H / 2 ? 1 : 2) : 0;
            initial += live;
        }
    }
    printf("initial live cells: %d of %d (seed %u)\n", initial, H * W, seed);

    std::vector<uint8_t> rgb;
    std::vector<uint8_t> grid = simulate(soup, n, steps, true, &rgb);

    int live = 0;
    for (uint8_t v : grid)
        live += (v != 0);
    printf("live cells after %d steps: %d (%.2f%%)\n", steps, live, 100.0 * live / (H * W));

    const int out_w = W / 4, out_h = H / 4;
    write_ppm("frame.ppm", rgb.data(), out_w, out_h);
    printf("wrote frame.ppm (%dx%d)\n", out_w, out_h);

    // The two-band run must produce exactly the one-band grid: same soup,
    // same steps, compared byte for byte. A difference means a halo index is
    // wrong, and this finds it before any NCCL is involved.
    if (check) {
        if (n == 1) {
            printf("check: only one band, nothing to compare against\n");
        } else {
            std::vector<uint8_t> ref = simulate(soup, 1, steps, false, nullptr);
            if (memcmp(grid.data(), ref.data(), grid.size()) == 0) {
                printf("check: %d-band grid identical to 1-band grid after %d steps\n", n, steps);
            } else {
                size_t i = 0;
                while (grid[i] == ref[i]) i++;
                fprintf(stderr, "check: MISMATCH at row %zu col %zu: %d bands give %d, 1 band gives %d\n",
                        i / W, i % W, n, grid[i], ref[i]);
                return 1;
            }
        }
    }

    return 0;
}
