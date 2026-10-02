# life_wars

Two-colour Game of Life on the GPU: red (1) and blue (2) cells follow the
normal Life rules, and a newborn cell takes the majority colour of its three
neighbours.

## Done so far

- **Grid:** 2048 x 2048, one byte per cell. The device buffer has a zeroed halo
  row above and below, so the grid is copied to `cur + W`. Columns wrap around.
- **`life_step` kernel:** one thread per cell, reads `cur`, writes `next`; the
  host swaps the pointers after each step.
- **Main:** 30% random soup (red top half, blue bottom half), 100 steps, live
  count by colour printed every 10 steps, then `cudaDeviceSynchronize()` to
  catch kernel faults.
- **`downsample` kernel + `write_ppm`:** renders the final grid at 4 x 4 cells
  per pixel and writes a 512 x 512 `frame.ppm`. Each rank shrinks its own
  band on its own GPU. Rank 1's strip reaches rank 0 through one more
  `ncclSend`/`ncclRecv` pair in a group, queued behind the `downsample`
  kernels on the same streams. Rank 0 then copies its own strip and the
  received one into the host image, top band first, and writes the file.
- **`--mark-bands`:** draws the first pixel row of GPU 1's band as a yellow
  seam line, so the split between the GPUs' strips is visible in every frame.
  It covers one row of pixels, which is why it is off by default.
- **`--full-size`:** writes `frame.ppm` and the `frames/` images at one cell
  per pixel (2048 x 2048) instead of 4 x 4 cells per pixel. The same
  `downsample` kernel runs with a shrink factor of 1, so a live cell is a
  full-brightness red or blue pixel.
- **Bands, one per GPU:** `--gpus N` (1 or 2) splits the grid into `n`
  horizontal bands, band `r` on GPU `r`. Band `r` owns `rows[r]` rows starting
  at `row0[r]` and has its own `cur`/`nxt` buffers with a halo row above and
  below. Every CUDA call for band `r` is made after `cudaSetDevice(r)`, and its
  kernels launch on `streams[r]`, one stream per GPU. The stream is not for
  overlap: NCCL calls take a stream, so the kernel and the NCCL work for one
  GPU go into the same queue and run in order without the host waiting.
- **NCCL:** `ncclCommInitAll` makes one communicator per GPU, rank `r` on
  GPU `r`, after the buffers are allocated; both are destroyed at the end.
- **Halo exchange:** before every step, a `ncclGroupStart`/`ncclGroupEnd`
  group holds four calls: rank 0 sends its last owned row to rank 1 and
  receives into its bottom halo; rank 1 sends its first owned row to rank 0
  and receives into its top halo. Each call goes on that rank's stream. The
  group matters because one host thread issues all four: outside a group the
  first send could wait for a receive the thread has not issued yet and hang.
  No synchronisation is needed before the kernel: the receive and the kernel
  sit on the same stream, and the GPU runs stream work in order, so the kernel
  starts only when the halo has arrived. Copying after the step instead would
  leave the first step blind. Everything two-GPU is behind `if (n > 1)`; with
  `--gpus 1` it is the single-buffer program as before.
- **`count_cells` kernel + all-reduce:** every 10 steps each band counts its
  red and blue cells into two 64-bit counters on its own GPU. A block tallies
  in shared memory first, then one thread per block does the `atomicAdd` to
  the global counters, so there are thousands of global atomics instead of
  millions. `ncclAllReduce` with `ncclSum` then adds the two ranks' counters
  in place: every rank puts in its two numbers and every rank gets the same
  two totals back. The host reads rank 0's after a `cudaStreamSynchronize`.
- **`--frame-every N`:** also renders the grid at step 0 and after every Nth
  step, through the same per-rank downsample and send/recv path, into
  `frames/frame_00000.ppm`, `frame_00001.ppm`, ... The index counts frames,
  not steps, so the sequence has no gaps and `make_video.sh` can turn it into
  an MP4 and a GIF. The folder is created if missing; `make clean` removes it.
- **Timing:** a `cudaEvent` is recorded on each GPU's stream before and after
  the step loop; `cudaEventElapsedTime` gives the milliseconds between the two
  marks on that stream, which includes the time the stream spent waiting for
  the other GPU's halo rows, not just its own kernels. `now_sec()` around the
  loop gives host wall time, printed as steps per second. With an equal split
  the two-GPU run is faster than one GPU but not twice as fast: the RTX 2070
  finishes its half later than the RTX 5070 Ti, and every step waits for both.
  The per-10-step count readback and `--frame-every` stall the host, so turn
  them off when comparing numbers.
- **`--check`:** with `--gpus 2`, also runs the same soup as one band and
  `memcmp`s the two final grids. A mismatch prints the first differing cell and
  exits 1; a halo index bug shows up here, before NCCL enters the picture.

## Build and run

Needs `nvcc` and one or two NVIDIA GPUs.

```sh
make run                          # runs ./life_wars --gpus 1
./life_wars --gpus 2 --check      # one band per GPU, verified against one band
./life_wars --seed 7 --steps 250  # other soups and lengths
./life_wars --gpus 2 --steps 600 --frame-every 2 && ./make_video.sh   # frames/ -> life_wars.mp4
```

A 30% soup should settle to a few percent alive within 100 steps.
