#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define MAX_NEIGHBORS 64  // Assume K <= MAX_NEIGHBORS for simplicity

// Enhanced CUDA kernel for neighborhood aggregation without warp reduction.
__global__ void enhanced_neighborhood_aggregation_kernel(
    const float* __restrict__ points,    // (B, N, C)
    const int* __restrict__ neighbors,     // (B, N, K)
    float* __restrict__ aggregated,        // (B, N, C)
    int B, int N, int K, int C)             // Dimensions: Batch, Points, Neighbors, Channels
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N;
    if (idx >= total)
        return;

    int b = idx / N;
    int n = idx % N;

    // Each thread computes the aggregation for one point over its K neighbors:
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            int neighbor_idx = neighbors[b * N * K + n * K + k];
            float val = points[b * N * C + neighbor_idx * C + c];
            sum += val;
        }
        aggregated[b * N * C + n * C + c] = sum / float(K);
    }
}

// Wrapper function exposed to Python.
torch::Tensor enhanced_neighborhood_aggregation(
    torch::Tensor points,    // (B, N, C)
    torch::Tensor neighbors, // (B, N, K)
    int C)                   // Number of channels
{
    auto B = points.size(0);
    auto N = points.size(1);
    auto K = neighbors.size(2);
    auto aggregated = torch::zeros({B, N, C}, points.options());

    int total = B * N;
    int threads = THREADS_PER_BLOCK;
    int blocks = (total + threads - 1) / threads;

    // Launch kernel without dynamic shared memory.
    enhanced_neighborhood_aggregation_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        neighbors.data_ptr<int>(),
        aggregated.data_ptr<float>(),
        B, N, K, C
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return aggregated;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enhanced_neighborhood_aggregation", &enhanced_neighborhood_aggregation, "Enhanced Neighborhood Aggregation CUDA Kernel");
}