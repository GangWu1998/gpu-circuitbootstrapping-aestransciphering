# GPU Circuit Bootstrapping AES Transciphering

This repository contains the AES-128 transciphering example used for the paper artifact.

Requirements:

- CUDA-capable NVIDIA GPU and CUDA toolkit.
- CMake 3.20 or newer.
- NVIDIA MathDx/cuFFTDx installed system-wide, under `thirdparty/nvidia-mathdx`, or with `MATHDX_ROOT` set.

Build:

```bash
git submodule update --init --recursive
cmake -S . -B build
cmake --build build --target example_aes128_transcipher -j
```

Run:

```bash
./build/bin/example_aes128_transcipher
```
