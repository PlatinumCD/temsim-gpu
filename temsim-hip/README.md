
# Temsim-hip

This directory will store the contents of Temsim using HIP

## Starting Point (Michael)

Tip: Pay attention to the Makefile, it may need adjusting

Steps:
1. Begin by making a branch on this repo (hip-port)
2. Copy the contents of `temsim-gpu/temsim-cuda` into this directory
3. Install HIPIFY
4. Practice using HIPIFY on basic CUDA cuda. Make sure the converted code runs on an AMD GPU on
   an NIC system. (Reach out to Erik if you need resources/permissions)
5. Build and run `temsim-gpu/temsim-cuda` on a NVIDIA GPU to understand the build process. (I will work on making sure `temsim-gpu/temsim-cuda` has input files and a good README)
6. Port Temsim to HIP and make sure you get same output (optimize later)
7. Make a pull request to main
8. Optimize performance
