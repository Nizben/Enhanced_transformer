import torch
import neighborhood_aggregation  # Import the compiled CUDA extension

def enhanced_neighborhood_aggregate(points, neighbors):
    """
    Wraps the custom CUDA kernel for neighborhood aggregation.
    Args:
        points (torch.Tensor): Input features (B, N, C).
        neighbors (torch.Tensor): Neighbor indices (B, N, K).
    Returns:
        torch.Tensor: Aggregated features (B, N, C).
    """
    # Ensure tensors are contiguous and on the correct device/type.
    points = points.contiguous().to(torch.float32)
    neighbors = neighbors.contiguous().to(torch.int32)
    
    aggregated = neighborhood_aggregation.enhanced_neighborhood_aggregation(points, neighbors, points.size(2))
    return aggregated
