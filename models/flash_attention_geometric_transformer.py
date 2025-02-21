import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func

class FlashAttentionGeometricTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(FlashAttentionGeometricTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scaling
        attn_output = flash_attn_varlen_func(
            q, k, v, attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.out_proj(attn_output)
        return self.dropout(attn_output)