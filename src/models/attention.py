"""Attention mechanism variants for Transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA).
    
    All query heads have their own key and value heads.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for causal mask
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (not a parameter)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: (batch, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose to (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use PyTorch 2.0's scaled_dot_product_attention with causal masking
        # This is optimized and memory-efficient
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True  # Automatically applies causal mask
        )
        
        # Reshape back: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA).
    
    All query heads share a single key and value head.
    This reduces memory usage during inference (smaller KV cache).
    
    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for causal mask
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q projection for all heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Single K and V projections (shared across all query heads)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q (multiple heads)
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        
        # Project to K, V (single head, will be broadcast)
        k = self.k_proj(x)  # (batch, seq_len, head_dim)
        v = self.v_proj(x)  # (batch, seq_len, head_dim)
        
        # Add head dimension and broadcast: (batch, 1, seq_len, head_dim)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        
        # Expand to match number of query heads: (batch, n_heads, seq_len, head_dim)
        k = k.expand(batch_size, self.n_heads, seq_len, self.head_dim)
        v = v.expand(batch_size, self.n_heads, seq_len, self.head_dim)
        
        # Scaled dot-product attention with causal masking
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA).
    
    Groups of query heads share key and value heads.
    This is a middle ground between MHA and MQA.
    
    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (must divide n_heads)
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for causal mask
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = self.head_dim * n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q projection for all heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # K, V projections for KV heads
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        
        # Project to K, V
        k = self.k_proj(x)  # (batch, seq_len, kv_dim)
        v = self.v_proj(x)
        
        # Reshape to separate KV heads
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        k = k.transpose(1, 2)  # (batch, n_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)
        
        # Repeat KV heads to match number of Q heads
        # Each KV head is repeated n_groups times
        k = k.repeat_interleave(self.n_groups, dim=1)  # (batch, n_heads, seq_len, head_dim)
        v = v.repeat_interleave(self.n_groups, dim=1)
        
        # Scaled dot-product attention with causal masking
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


def create_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    dropout: float = 0.1,
    max_seq_len: int = 2048
) -> nn.Module:
    """Factory function to create attention modules.
    
    Args:
        attention_type: Type of attention ("mha", "mqa", "gqa")
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (for GQA)
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
        
    Returns:
        Attention module
    """
    if attention_type == "mha":
        return MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
    elif attention_type == "mqa":
        return MultiQueryAttention(d_model, n_heads, dropout, max_seq_len)
    elif attention_type == "gqa":
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 2)  # Default to half
        return GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, max_seq_len)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
