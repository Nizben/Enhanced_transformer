# Kernel-Enhanced Geometric Transformer for 3D Point Cloud Processing

## Overview

This project implements a high-performance 3D point cloud processing pipeline using a **Kernel-Enhanced Geometric Transformer**. It integrates:
- **KeOps** for efficient Gaussian kernel computations
- **Flash Attention** for scalable attention in Transformer layers
- **Custom CUDA Kernels** for highly optimized neighborhood aggregation

I am basically playing around with **KeOps**, **Flash Attention**, and a **customized CUDA kernel** that does neighborhood aggregation. The idea is for me to learn how to manipulate these things, combine them if needed, benchmark their performance, and try to experiment with this unconventional transformer.

> Note : Each block of this transformer architecture (the 3 blocks above) is highly optimized, but their combination and the architecture as a whole is not necessarly. This is more of an educational project.