"""Transformer block implementation."""

import torch
import torch.nn as nn
from typing import Tuple


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture.
    
    Architecture (pre-norm):
        x = x + attn(LayerNorm(x))
        x = x + ffn(LayerNorm(x))
    
    Args:
        attention: Attention module (MHA, MQA, or GQA)
        ffn: Feed-forward network module (Dense or MoE)
        d_model: Model dimension
        dropout: Residual dropout probability
    """
    
    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        d_model: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        
        # Layer normalization (pre-norm)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tuple of (output, aux_loss) where aux_loss is from FFN (0 for dense)
        """
        # Self-attention with residual connection (pre-norm)
        attn_output = self.attention(self.ln1(x))
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection (pre-norm)
        ffn_output, aux_loss = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)
        
        return x, aux_loss
