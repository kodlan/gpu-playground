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
  per pixel and writes a 512 x 512 `frame.ppm`. Each band renders its own
  strip on its own GPU; the host stitches the strips together.
- **Bands, one per GPU:** `--gpus N` (1 or 2) splits the grid into `n`
  horizontal bands, band `r` on GPU `r`. Band `r` owns `rows[r]` rows starting
  at `row0[r]` and has its own `cur`/`nxt` buffers with a halo row above and
  below. Every CUDA call for band `r` is made after `cudaSetDevice(r)`, and its
  kernels launch on `streams[r]`, one stream per GPU. The stream is not for
  overlap: NCCL calls take a stream, so the kernel and the NCCL work for one
  GPU go into the same queue and run in order without the host waiting.
- **Halo exchange:** before every step, `cudaMemcpyPeer` copies band 0's last
  row (GPU 0) into band 1's top halo (GPU 1) and band 1's first row into band
  0's bottom halo. Without peer-to-peer access the driver stages the copy
  through host memory. The kernel reads the halos, so copying after the step
  would leave the first step blind. Everything two-GPU is behind `if (n > 1)`;
  with `--gpus 1` it is the single-buffer program as before.
- **`--check`:** with `--gpus 2`, also runs the same soup as one band and
  `memcmp`s the two final grids. A mismatch prints the first differing cell and
  exits 1; a halo index bug shows up here, before NCCL enters the picture.

## Build and run

Needs `nvcc` and one or two NVIDIA GPUs.

```sh
make run                          # runs ./life_wars --gpus 1
./life_wars --gpus 2 --check      # one band per GPU, verified against one band
./life_wars --seed 7 --steps 250  # other soups and lengths
```

A 30% soup should settle to a few percent alive within 100 steps.
