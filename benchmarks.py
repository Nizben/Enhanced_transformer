import torch
from models.custom_cuda_neighborhood import enhanced_neighborhood_aggregation

def benchmark_pytorch(points, neighbors, num_iterations=100, warmup=10):
    """
    Benchmark pure PyTorch neighborhood aggregation using advanced indexing and torch.mean.
    Returns average time per iteration (ms) and one sample output.
    """
    B, N, C = points.shape
    K = neighbors.shape[-1]
    # Create batch indices for advanced indexing.
    batch_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand(B, N, K)

    # Warmup runs
    for _ in range(warmup):
        # points shape: (B, N, C) and neighbors: (B, N, K) 
        # gathered neighbor_features: (B, N, K, C)
        neighbor_features = points[batch_idx, neighbors]
        _ = neighbor_features.mean(dim=2)
    torch.cuda.synchronize()
    
    # One sample run for output validation
    neighbor_features = points[batch_idx, neighbors]
    output = neighbor_features.mean(dim=2)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        neighbor_features = points[batch_idx, neighbors]
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
    # Warmup runs
    for _ in range(warmup):
        output = custom_cuda_function(points, neighbors)
    torch.cuda.synchronize()
    
    output = custom_cuda_function(points, neighbors)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        _ = custom_cuda_function(points, neighbors)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / num_iterations
    return elapsed_time, output

if __name__ == "__main__":
    # Parameters
    B, N, C, K = 4, 1024, 64, 16
    num_iterations = 100
    warmup = 10

    # Generate random test data (on CUDA).
    points = torch.rand(B, N, C, device='cuda')
    neighbors = torch.randint(0, N, (B, N, K), device='cuda')

    pytorch_time, pytorch_output = benchmark_pytorch(points, neighbors, num_iterations, warmup)
    print("Pure PyTorch average time per iteration: {:.4f} ms".format(pytorch_time))

    cuda_time, cuda_output = benchmark_cuda_kernel(
        points, neighbors, num_iterations, warmup,
        custom_cuda_function=enhanced_neighborhood_aggregation
    )
    print("Enhanced CUDA kernel average time per iteration: {:.4f} ms".format(cuda_time))

    if not torch.allclose(pytorch_output, cuda_output, rtol=1e-05, atol=1e-06):
        max_diff = (pytorch_output - cuda_output).abs().max().item()
        raise ValueError(f"Outputs differ! Maximum absolute difference: {max_diff}")
    else:
        print("Output validation passed: Both implementations produce nearly identical results.")

    speedup = pytorch_time / cuda_time
    print("Speedup of CUDA kernel over PyTorch: {:.2f}x".format(speedup))
