import torch
import torch.nn as nn
from .kernel_geometric_operations import KernelDistance
from .flash_attention_geometric_transformer import FlashAttentionGeometricTransformerLayer
from .custom_cuda_neighborhood import enhanced_neighborhood_aggregation

class EnhancedGeometricTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, sigma=1.0, dropout=0.1, K=16, num_classes=10):
        super(EnhancedGeometricTransformer, self).__init__()
        self.kernel_distance = KernelDistance(sigma=sigma)
        self.transformer_layers = nn.ModuleList([
            FlashAttentionGeometricTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, points, neighbors):
        # Compute Gaussian kernel similarity matrix: (B, N, N)
        K_sim = self.kernel_distance(points)
        # Aggregate features using the custom enhanced CUDA kernel: (B, N, embed_dim)
        aggregated_features = enhanced_neighborhood_aggregation(K_sim, neighbors, points.size(2))
        x = aggregated_features
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        x = torch.max(x, dim=1)[0]
        logits = self.fc_out(x)
        return logits