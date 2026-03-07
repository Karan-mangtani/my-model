"""Feed-forward network variants for Transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DenseFFN(nn.Module):
    """Standard dense feed-forward network.
    
    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tuple of (output, aux_loss) where aux_loss is 0 for dense FFN
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Return zero auxiliary loss for compatibility with MoE
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        return x, aux_loss


class MoEFFN(nn.Module):
    """Mixture of Experts Feed-Forward Network.
    
    Each token is routed to top-k experts based on a learned gating network.
    Includes load balancing loss to encourage uniform expert usage.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension for each expert
        num_experts: Number of expert networks
        top_k: Number of experts to route each token to
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        assert 0 < top_k <= num_experts, f"top_k must be between 1 and {num_experts}"
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network: produces routing logits for each expert
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks (each is a DenseFFN)
        self.experts = nn.ModuleList([
            self._create_expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_expert(self, d_model: int, d_ff: int, dropout: float) -> nn.Module:
        """Create a single expert network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            
        Returns:
            Expert module (a simple FFN)
        """
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with top-k routing.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tuple of (output, aux_loss) where aux_loss is the load balancing loss
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape to (batch * seq_len, d_model) for easier processing
        x_flat = x.view(-1, d_model)
        
        # Compute routing logits: (batch * seq_len, num_experts)
        router_logits = self.gate(x_flat)
        
        # Compute routing probabilities and select top-k experts
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Route tokens to experts and combine outputs
        for k in range(self.top_k):
            # Get expert indices for this k
            expert_indices = top_k_indices[:, k]  # (batch * seq_len,)
            expert_weights = top_k_probs[:, k:k+1]  # (batch * seq_len, 1)
            
            # Process each expert
            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                mask = (expert_indices == expert_id)
                
                if mask.any():
                    # Get tokens for this expert
                    expert_input = x_flat[mask]
                    
                    # Process through expert
                    expert_output = self.experts[expert_id](expert_input)
                    
                    # Weight by routing probability and add to output
                    expert_weights_masked = expert_weights[mask]
                    output[mask] += expert_weights_masked * expert_output
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute load balancing loss
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.
        
        Encourages uniform distribution of tokens across experts.
        
        Args:
            router_probs: Router probabilities (num_tokens, num_experts)
            top_k_indices: Top-k expert indices (num_tokens, top_k)
            
        Returns:
            Load balancing loss scalar
        """
        num_tokens = router_probs.shape[0]
        
        # Compute fraction of tokens assigned to each expert
        expert_counts = torch.zeros(self.num_experts, device=router_probs.device)
        for expert_id in range(self.num_experts):
            expert_counts[expert_id] = (top_k_indices == expert_id).float().sum()
        
        expert_fractions = expert_counts / (num_tokens * self.top_k)
        
        # Compute average routing probability for each expert
        avg_router_probs = router_probs.mean(dim=0)
        
        # Load balancing loss: encourages uniform expert usage
        # This is the auxiliary loss from Switch Transformer paper
        # loss = num_experts * sum(expert_fraction * avg_prob)
        aux_loss = self.num_experts * (expert_fractions * avg_router_probs).sum()
        
        return aux_loss


def create_ffn(
    ffn_type: str,
    d_model: int,
    d_ff: int,
    num_experts: int = 8,
    top_k: int = 2,
    dropout: float = 0.1
) -> nn.Module:
    """Factory function to create FFN modules.
    
    Args:
        ffn_type: Type of FFN ("dense" or "moe")
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_experts: Number of experts (for MoE)
        top_k: Number of experts per token (for MoE)
        dropout: Dropout probability
        
    Returns:
        FFN module
    """
    if ffn_type == "dense":
        return DenseFFN(d_model, d_ff, dropout)
    elif ffn_type == "moe":
        return MoEFFN(d_model, d_ff, num_experts, top_k, dropout)
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")
