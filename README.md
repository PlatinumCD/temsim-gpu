# TEMSIM-GPU

temsim-gpu is a collection of GPU-accelerated simulations for computational fluid dynamics (CFD) and transmission electron microscopy (TEM). This project includes implementations using CUDA, HIP, and SYCL for different hardware platforms.

## Implementations

- temsim: CPU implementation.
- temsim-cuda: CUDA-based implementation for NVIDIA GPUs.
- temsim-hip: HIP-based implementation for AMD and NVIDIA GPUs.
- temsim-sycl: SYCL-based implementation for cross-platform support.

Each implementation has its own README.md with further instructions on setup and usage.

## Installation

### `temsim`

Build `temsim` using the script `build_temsim.sh`:
```bash
./build_temsim.sh
```

This file will install FFTW locally and build temsim for the CPU. The executables will be located at `temsim/build`.

**Tested on NIC Saturn**

### `temsim-cuda`

Build `temsim-cuda` using the script `build_temsim-cuda.sh`:
```bash
./build_temsim-cuda.sh
```

This file will install FFTW locally and build temsim for NVIDIA GPUs (assuming compute capability 80). The executables will be located at `temsim-cuda/build`.

**Tested on NIC Saturn**

## License

Licensing details are pending and will be updated soon.

Stay tuned for updates!
