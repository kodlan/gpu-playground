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
  `make_video.sh --crop square` or `--crop vertical` cuts a window centred on
  the colour border out of every frame, which is the simple way to zoom in on
  the ships. Square output is the window itself; vertical output is always
  720 x 900 px, cut as a 4:5 window and scaled without smoothing. `--size PX`
  sets the window width (default a quarter of the frame height for square,
  720 for vertical on full-size frames, so that video is one cell per pixel)
  and `--center X` moves the window sideways to frame column X, for example
  onto a gun. Use it with `--full-size` frames so a column is a cell.
- **Timing:** a `cudaEvent` is recorded on each GPU's stream before and after
  the step loop; `cudaEventElapsedTime` gives the milliseconds between the two
  marks on that stream, which includes the time the stream spent waiting for
  the other GPU's halo rows, not just its own kernels. `now_sec()` around the
  loop gives host wall time, printed as steps per second. With an equal split
  the two-GPU run is faster than one GPU but not twice as fast: the RTX 2070
  finishes its half later than the RTX 5070 Ti, and every step waits for both.
  The per-10-step count readback and `--frame-every` stall the host, so turn
  them off when comparing numbers.
- **`--balance`:** `rows_per_sec(dev, W)` runs `life_step` on a 1024-row slab
  40 times on each GPU and times it with events (after one untimed warm-up
  launch). The grid is then split in proportion, `rows[0] = H * r0 / (r0 +
  r1)`, rounded down to a multiple of the shrink factor so both strips still
  render whole pixels. The measured rates and the split are printed.
- **`--ships`** (or `--gliders`): `make_grid` clears a strip of `H / 32`
  rows on each side of the colour border (row `H / 2`) and launches one ship
  per `W / 32` columns from each edge of the strip, pointing at the border:
  red ships fly down from the red soup, blue ships fly up from the blue soup.
  The type and the column inside each lane are random (same seed as the
  soup), so some pairs meet head-on at the border and others pass and fly
  into the opposite soup, where their colour wins the births and seeds a
  colony on the wrong side. Five ships are in the table, each stored in its
  down-going orientation and flipped for the blue side: the glider (5 cells,
  c/4 diagonal), the heavyweight spaceship HWSS (13 cells, c/2), the loafer
  (20 cells, c/7), the copperhead (28 cells, c/10) and Puffer 1 (25 cells,
  c/2), a B-heptomino escorted by two LWSS that leaves a trail of debris of
  its own colour behind it. Every `W / 4` columns a Gosper glider gun (36
  cells) takes the place of the ships in its lanes and fires a glider at the
  border every 30 steps for as long as it survives; a notch cleared into the
  soup behind each gun keeps the soup's edge debris off it. Red guns sit at
  columns `W / 8 + k * W / 4` (centre of gun k). The blue gun of every even
  stretch sits in the same columns, so its stream meets the red one head-on
  at the border and the guns duel; the blue gun of every odd stretch is
  shifted by `W / 8`, so both streams fly past each other into the enemy
  soup. The program prints these positions. Every layout was checked
  in a small Life simulator before going into the table. The HWSS reaches the
  border in about 110 steps and the copperhead in about 560, so use
  `--steps 800` or more with `--frame-every` to watch it all.
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
./life_wars --gpus 2 --steps 800 --ships --frame-every 2 && ./make_video.sh   # spaceships collide at the border
./life_wars --gpus 2 --steps 800 --ships --full-size --frame-every 4 && ./make_video.sh --crop square     # zoom on the border
./make_video.sh --crop vertical frames strip.mp4 30                              # 720 x 900 window, one cell per pixel
./make_video.sh --crop square --center 320 frames duel.mp4      # gun 0: red and blue streams meet at the border
./make_video.sh --crop square --center 900 frames raid.mp4      # gun 1: red gun at 768, blue gun at 1024, streams get through
```

A 30% soup should settle to a few percent alive within 100 steps.
