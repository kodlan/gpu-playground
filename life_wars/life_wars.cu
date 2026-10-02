#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "../common/check.h"

static const int W = 2048, H = 2048;
static const int SHRINK = 4;              // rendered frames: SHRINK x SHRINK cells per pixel (1 with --full-size)
static const char* FRAMES_DIR = "frames"; // --frame-every writes frames/frame_00000.ppm, ...


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

// Count the band's red and blue cells into counts[0] and counts[1]. Each block
// first tallies in shared memory (fast, on-chip), then one thread per block
// adds the block's totals to the global counters with atomicAdd: hundreds of
// atomic adds to global memory per call instead of millions.
__global__ void count_cells(const uint8_t* __restrict__ cur, int rows, int columns, unsigned long long* counts) {
    __shared__ int s_red, s_blue;
    bool leader = (threadIdx.x == 0 && threadIdx.y == 0);

    if (leader) {
        s_red = 0;
        s_blue = 0;
    }
    __syncthreads();

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < columns && r < rows) {
        uint8_t v = cur[(r + 1) * columns + c]; // +1 skips the top halo row
        if (v == 1)
            atomicAdd(&s_red, 1);
        else if (v == 2)
            atomicAdd(&s_blue, 1);
    }
    __syncthreads();

    if (leader) {
        if (s_red)
            atomicAdd(&counts[0], (unsigned long long)s_red);
        if (s_blue)
            atomicAdd(&counts[1], (unsigned long long)s_blue);
    }
}

// Shrink the band by `shrink` in each direction into an RGB image: each output
// pixel shows the colour that has more cells in its shrink x shrink block.
// With `mark` set (--mark-bands) the first pixel row of every band after the
// first is drawn as a yellow seam line, so the strip each GPU drew is visible.
__global__ void downsample(const uint8_t* __restrict__ cur, int rows, int colums, int shrink, uint8_t* rgb, int out_w, int band, int mark) {
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

    // seam line: the first pixel row of band 1 (and later bands)
    if (mark && band > 0 && oy == 0) {
        p[0] = 230;
        p[1] = 200;
        p[2] = 40;
        return;
    }

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
// final grid is rendered into it at shrink x shrink cells per pixel. With
// `frame_every` > 0 the grid is also rendered at step 0 and after every
// frame_every-th step, each frame going to FRAMES_DIR/frame_NNNNN.ppm with a
// running index, which is what make_video.sh expects. `mark_bands` makes the
// strip each GPU rendered visible (see downsample).
static std::vector<uint8_t> simulate(const std::vector<uint8_t>& init, int n, int steps, bool verbose, std::vector<uint8_t>* rgb, int frame_every, bool mark_bands, int shrink) {
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

    // per-band counters: counts_d[r][0] red, counts_d[r][1] blue, on GPU r
    unsigned long long* counts_d[2] = {nullptr, nullptr};
    for (int r = 0; r < n; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&counts_d[r], 2 * sizeof(unsigned long long)));
    }

    // one NCCL communicator per GPU, all in this process: ncclCommInitAll
    // sets up comms[i] on devs[i]. Only needed when there are two bands to talk.
    ncclComm_t comms[2] = {nullptr, nullptr};
    if (n > 1) {
        int devs[2] = {0, 1};
        NCCL_CHECK(ncclCommInitAll(comms, 2, devs));
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

    // Rendering. Each rank shrinks its own band on its own GPU into a strip
    // of out_w x (rows[r] / shrink) pixels, straight from the device buffer
    // (downsample skips the halo row itself, so pass cur, not cur + W).
    // rows[r] is a multiple of shrink for H = 2048, n <= 2 and shrink 1 or 4.
    // The strips and rank 0's receive buffer are allocated once and reused
    // for every frame.
    const bool rendering = (rgb != nullptr) || frame_every > 0;
    const int out_w = W / shrink, out_h = H / shrink;
    uint8_t* d_rgb[2] = {nullptr, nullptr};
    size_t strip_bytes[2] = {0, 0};
    uint8_t* d_recv = nullptr;  // rank 1's strip, landed on GPU 0
    if (rendering) {
        for (int r = 0; r < n; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            strip_bytes[r] = (size_t)out_w * (rows[r] / shrink) * 3;
            CUDA_CHECK(cudaMalloc(&d_rgb[r], strip_bytes[r]));
        }
        if (n > 1) {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMalloc(&d_recv, strip_bytes[1]));
        }
    }

    // render the current grid into the host image `out`
    auto render = [&](std::vector<uint8_t>& out) {
        out.resize((size_t)out_w * out_h * 3);
        for (int r = 0; r < n; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            int band_h = rows[r] / shrink;
            dim3 og((out_w + block.x - 1) / block.x, (band_h + block.y - 1) / block.y);
            downsample<<<og, block, 0, streams[r]>>>(cur[r], rows[r], W, shrink, d_rgb[r], out_w, r, mark_bands ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
        }

        // Rank 1's small image has to reach rank 0, which writes the file:
        // one more send/recv pair in a group. Both calls sit behind the
        // downsample kernels on their streams, so the strips are finished
        // when the bytes move.
        if (n > 1) {
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclSend(d_rgb[1], strip_bytes[1], ncclUint8, 0, comms[1], streams[1]));  // rank 1 sends its strip to rank 0
            NCCL_CHECK(ncclRecv(d_recv,   strip_bytes[1], ncclUint8, 1, comms[0], streams[0]));  // rank 0 receives it
            NCCL_CHECK(ncclGroupEnd());
        }

        // rank 0 copies its own strip and then the received one into the host
        // image, top band first; the blocking copies wait for streams[0]
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(out.data(), d_rgb[0], strip_bytes[0], cudaMemcpyDeviceToHost));
        if (n > 1)
            CUDA_CHECK(cudaMemcpy(out.data() + strip_bytes[0], d_recv, strip_bytes[1], cudaMemcpyDeviceToHost));
    };

    // frames/frame_00000.ppm, frame_00001.ppm, ... numbered by frame, not by
    // step, so the sequence has no gaps for ffmpeg
    int frames_written = 0;
    std::vector<uint8_t> frame;
    auto write_frame = [&]() {
        render(frame);
        char path[64];
        snprintf(path, sizeof path, "%s/frame_%05d.ppm", FRAMES_DIR, frames_written);
        write_ppm(path, frame.data(), out_w, out_h);
        frames_written++;
    };
    if (frame_every > 0) {
        if (mkdir(FRAMES_DIR, 0755) != 0 && errno != EEXIST) {
            perror(FRAMES_DIR);
            exit(1);
        }
        write_frame();  // step 0: the soup
    }

    for (int s = 0; s < steps; s++) {
        // Halo exchange BEFORE the step: the kernel reads the halo rows, so
        // they must hold the neighbour's current edge row when it runs. The
        // first step would otherwise see empty halos and the bands drift apart.
        //
        // Each line reads "rank X sends/receives W bytes to/from rank Y on its
        // stream"; the pointer says where the bytes come from or go to.
        //
        // Why the group: this one thread issues all four calls. Without it the
        // first ncclSend could wait for rank 1's matching ncclRecv, which this
        // thread has not issued yet, and the program would hang forever. The
        // group tells NCCL "collect these, then start them all together".
        //
        // Why no synchronisation before the kernel: the receive is queued on
        // streams[r] and the kernel is queued on the same stream right after
        // it. The GPU runs stream work in order, so the kernel starts only
        // when the halo has arrived.
        if (n > 1) {
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclSend(cur[0] + (size_t)rows[0] * W,       W, ncclUint8, 1, comms[0], streams[0]));  // rank 0's last owned row
            NCCL_CHECK(ncclRecv(cur[0] + (size_t)(rows[0] + 1) * W, W, ncclUint8, 1, comms[0], streams[0]));  // into rank 0's bottom halo
            NCCL_CHECK(ncclSend(cur[1] + W,                         W, ncclUint8, 0, comms[1], streams[1]));  // rank 1's first owned row
            NCCL_CHECK(ncclRecv(cur[1],                             W, ncclUint8, 0, comms[1], streams[1]));  // into rank 1's top halo
            NCCL_CHECK(ncclGroupEnd());
        }

        for (int r = 0; r < n; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            dim3 g((W + block.x - 1) / block.x, (rows[r] + block.y - 1) / block.y);
            life_step<<<g, block, 0, streams[r]>>>(cur[r], nxt[r], rows[r], W);
            std::swap(cur[r], nxt[r]);
        }

        // Population count. Each band counts its own cells on its stream,
        // right behind the step kernel, so no host wait is needed in between.
        if (verbose && (s + 1) % 10 == 0) {
            for (int r = 0; r < n; r++) {
                CUDA_CHECK(cudaSetDevice(r));
                CUDA_CHECK(cudaMemsetAsync(counts_d[r], 0, 2 * sizeof(unsigned long long), streams[r]));
                dim3 g((W + block.x - 1) / block.x, (rows[r] + block.y - 1) / block.y);
                count_cells<<<g, block, 0, streams[r]>>>(cur[r], rows[r], W, counts_d[r]);
            }

            // Sum across GPUs. All-reduce: every rank puts in its two counts,
            // NCCL adds them up, and every rank gets the same two totals back.
            // Same buffer for input and output, so the reduction is in place.
            // One group again, because this thread issues both ranks' calls.
            if (n > 1) {
                NCCL_CHECK(ncclGroupStart());
                for (int r = 0; r < n; r++)
                    NCCL_CHECK(ncclAllReduce(counts_d[r], counts_d[r], 2, ncclUint64, ncclSum, comms[r], streams[r]));
                NCCL_CHECK(ncclGroupEnd());
            }

            // every rank has the totals, so read rank 0's. The stream wait
            // stalls the host, so this is for watching the population only;
            // drop it when timing.
            unsigned long long counts[2];
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpyAsync(counts, counts_d[0], sizeof counts, cudaMemcpyDeviceToHost, streams[0]));
            CUDA_CHECK(cudaStreamSynchronize(streams[0]));
            printf("step %3d: live %7llu (red %7llu, blue %7llu)\n", s + 1, counts[0] + counts[1], counts[0], counts[1]);
        }

        // every frame_every-th step goes to a file. The downsample kernels
        // queue behind the step kernels on the same streams, so this sees the
        // finished step; the copy to the host stalls, so drop it when timing.
        if (frame_every > 0 && (s + 1) % frame_every == 0)
            write_frame();
    }
    for (int r = 0; r < n; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaGetLastError());
        // an out-of-bounds read in a kernel surfaces here, not at the launch
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    gather();

    if (rgb)
        render(*rgb);
    if (frame_every > 0)
        printf("wrote %d frames to %s/frame_%05d.ppm .. frame_%05d.ppm (%dx%d)\n",
               frames_written, FRAMES_DIR, 0, frames_written - 1, out_w, out_h);

    if (rendering) {
        for (int r = 0; r < n; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaFree(d_rgb[r]));
        }
        if (n > 1) {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaFree(d_recv));
        }
    }

    for (int r = 0; r < n; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaFree(cur[r]));
        CUDA_CHECK(cudaFree(nxt[r]));
        CUDA_CHECK(cudaFree(counts_d[r]));
        CUDA_CHECK(cudaStreamDestroy(streams[r]));
    }
    if (n > 1) {
        for (int r = 0; r < n; r++)
            NCCL_CHECK(ncclCommDestroy(comms[r]));
    }
    return grid;
}

static void usage(const char* prog) {
    fprintf(stderr,
            "usage: %s [--gpus N] [--steps N] [--seed N] [--frame-every N] [--mark-bands] [--full-size] [--check]\n"
            "  --gpus N         number of GPUs, 1 or 2; the grid is split into one band per GPU\n"
            "  --steps N        generations to run (default 100)\n"
            "  --seed N         seed for the random soup (default 1)\n"
            "  --frame-every N  also write a frame at step 0 and after every Nth step\n"
            "                   to %s/frame_00000.ppm, frame_00001.ppm, ... (default off)\n"
            "  --mark-bands     draw a yellow seam line where the GPUs' bands meet\n"
            "  --full-size      write images at one cell per pixel instead of %d x %d cells per pixel\n"
            "  --check          with --gpus 2, also run one band and compare the grids\n",
            prog, FRAMES_DIR, SHRINK, SHRINK);
    exit(2);
}

int main(int argc, char** argv) {
    int n = 1;          // number of GPUs, one band each
    int steps = 100;
    unsigned seed = 1;
    int frame_every = 0;  // 0: only the final frame.ppm
    bool mark_bands = false;
    bool full_size = false;
    bool check = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpus" && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc)
            steps = atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc)
            seed = (unsigned)strtoul(argv[++i], nullptr, 10);
        else if (a == "--frame-every" && i + 1 < argc)
            frame_every = atoi(argv[++i]);
        else if (a == "--mark-bands")
            mark_bands = true;
        else if (a == "--full-size")
            full_size = true;
        else if (a == "--check")
            check = true;
        else
            usage(argv[0]);
    }
    if (n < 1 || n > 2) {
        fprintf(stderr, "--gpus must be 1 or 2\n");
        return 2;
    }
    if (frame_every < 0) {
        fprintf(stderr, "--frame-every must be positive\n");
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
    const int shrink = full_size ? 1 : SHRINK;
    std::vector<uint8_t> grid = simulate(soup, n, steps, true, &rgb, frame_every, mark_bands, shrink);

    int live = 0;
    for (uint8_t v : grid)
        live += (v != 0);
    printf("live cells after %d steps: %d (%.2f%%)\n", steps, live, 100.0 * live / (H * W));

    const int out_w = W / shrink, out_h = H / shrink;
    write_ppm("frame.ppm", rgb.data(), out_w, out_h);
    printf("wrote frame.ppm (%dx%d)\n", out_w, out_h);

    // The two-band run must produce exactly the one-band grid: same soup,
    // same steps, compared byte for byte. A difference means a halo index is
    // wrong, and this finds it before any NCCL is involved.
    if (check) {
        if (n == 1) {
            printf("check: only one band, nothing to compare against\n");
        } else {
            std::vector<uint8_t> ref = simulate(soup, 1, steps, false, nullptr, 0, false, shrink);
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
