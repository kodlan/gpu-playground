# life_wars

Two-colour Game of Life on the GPU: red (1) and blue (2) cells follow the
normal Life rules, and a newborn cell takes the majority colour of its three
neighbours.

## Done so far

- **Grid:** 512 x 512, one byte per cell. The device buffer has a zeroed halo
  row above and below, so the grid is copied to `cur + W`. Columns wrap around.
- **`life_step` kernel:** one thread per cell, reads `cur`, writes `next`; the
  host swaps the pointers after each step.
- **Main:** 30% random soup (red top half, blue bottom half), 100 steps, live
  count by colour printed every 10 steps, then `cudaDeviceSynchronize()` to
  catch kernel faults.
- **`downsample` kernel + `write_ppm`:** renders the final grid at 4 x 4 cells
  per pixel and writes a 128 x 128 `frame.ppm`.

## Build and run

Needs `nvcc` and an NVIDIA GPU.

```sh
make run
```

A 30% soup should settle to a few percent alive within 100 steps.
