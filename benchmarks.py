# benchmarking.py

import torch
from custom_cuda_neighborhood import enhanced_neighborhood_aggregate  # Adjust the import path as needed

def benchmark_pytorch(points, neighbors, num_iterations=100, warmup=10):
    """
    Benchmark pure PyTorch neighborhood aggregation using torch.gather and torch.mean.
    Returns average time per iteration (ms) and one sample output.
    """
    B, N, C = points.shape
    for _ in range(warmup):
        neighbor_features = torch.gather(points, 1, neighbors.unsqueeze(-1).expand(-1, -1, -1, C))
        aggregated = neighbor_features.mean(dim=2)
    torch.cuda.synchronize()
    
    # One sample run for output validation
    neighbor_features = torch.gather(points, 1, neighbors.unsqueeze(-1).expand(-1, -1, -1, C))
    output = neighbor_features.mean(dim=2)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        neighbor_features = torch.gather(points, 1, neighbors.unsqueeze(-1).expand(-1, -1, -1, C))
        _ = neighbor_features.mean(dim=2)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / num_iterations
    return elapsed_time, output

def benchmark_cuda_kernel(points, neighbors, num_iterations=100, warmup=10, custom_cuda_function=None):
    """
    Benchmark the custom CUDA kernel for neighborhood aggregation.
    Returns average time per iteration (ms) and one sample output.
    """
    for _ in range(warmup):
        output = custom_cuda_function(points, neighbors, points.size(2))
    torch.cuda.synchronize()
    
    output = custom_cuda_function(points, neighbors, points.size(2))
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        _ = custom_cuda_function(points, neighbors, points.size(2))
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / num_iterations
    return elapsed_time, output

if __name__ == "__main__":
    # Parameters
    B, N, C, K = 4, 1024, 64, 16
    num_iterations = 100
    warmup = 10

    # Generate random test data
    points = torch.rand(B, N, C, device='cuda')
    neighbors = torch.randint(0, N, (B, N, K), device='cuda')

    pytorch_time, pytorch_output = benchmark_pytorch(points, neighbors, num_iterations, warmup)
    print("Pure PyTorch average time per iteration: {:.4f} ms".format(pytorch_time))

    cuda_time, cuda_output = benchmark_cuda_kernel(points, neighbors, num_iterations, warmup, custom_cuda_function=enhanced_neighborhood_aggregate)
    print("Enhanced CUDA kernel average time per iteration: {:.4f} ms".format(cuda_time))

    if not torch.allclose(pytorch_output, cuda_output, rtol=1e-05, atol=1e-06):
        max_diff = (pytorch_output - cuda_output).abs().max().item()
        raise ValueError(f"Outputs differ! Maximum absolute difference: {max_diff}")
    else:
        print("Output validation passed: Both implementations produce nearly identical results.")

    speedup = pytorch_time / cuda_time
    print("Speedup of CUDA kernel over PyTorch: {:.2f}x".format(speedup))
