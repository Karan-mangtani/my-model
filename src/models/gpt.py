"""Decoder-only Transformer language model (GPT-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from .attention import create_attention
from .ffn import create_ffn
from .transformer import TransformerBlock


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.
    
    Architecture:
        - Token embeddings + learned positional embeddings
        - N Transformer blocks (attention + FFN)
        - Final layer norm
        - LM head (tied with token embeddings)
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
        attention_type: Type of attention ("mha", "mqa", "gqa")
        n_kv_heads: Number of key/value heads (for GQA)
        ffn_type: Type of FFN ("dense", "moe")
        num_experts: Number of experts (for MoE)
        top_k: Number of experts per token (for MoE)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        attention_type: str = "mha",
        n_kv_heads: int = None,
        ffn_type: str = "dense",
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Learned positional embeddings
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                attention=create_attention(
                    attention_type=attention_type,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len
                ),
                ffn=create_ffn(
                    ffn_type=ffn_type,
                    d_model=d_model,
                    d_ff=d_ff,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=dropout
                ),
                d_model=d_model,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # LM head (will be tied with token embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between token embeddings and LM head
        self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with GPT-style initialization."""
        # Token embeddings: small std
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        
        # Positional embeddings: small std
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        
        # Apply special initialization to residual projections
        # Scale by 1/sqrt(n_layers) for better deep network training
        scale_factor = 1.0 / math.sqrt(self.n_layers)
        
        for block in self.blocks:
            # Scale attention output projection
            if hasattr(block.attention, 'out_proj'):
                nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=0.02 * scale_factor)
            
            # Scale FFN output projection
            if hasattr(block.ffn, 'fc2'):
                # Dense FFN
                nn.init.normal_(block.ffn.fc2.weight, mean=0.0, std=0.02 * scale_factor)
            elif hasattr(block.ffn, 'experts'):
                # MoE FFN - scale each expert's output projection
                for expert in block.ffn.experts:
                    if hasattr(expert, '__getitem__'):
                        # Expert is nn.Sequential, last linear is at index -2 (before dropout)
                        nn.init.normal_(expert[-2].weight, mean=0.0, std=0.02 * scale_factor)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            targets: Target token IDs of shape (batch, seq_len) for loss computation
            
        Returns:
            Tuple of (logits, lm_loss, aux_loss) where:
                - logits: (batch, seq_len, vocab_size)
                - lm_loss: Cross-entropy loss (or None if targets not provided)
                - aux_loss: Auxiliary loss from MoE (0 for dense models)
        """
        batch_size, seq_len = input_ids.shape
        
        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)  # (batch, seq_len, d_model)
        
        # Get positional embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embeddings(positions)  # (batch, seq_len, d_model)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Pass through transformer blocks and accumulate auxiliary loss
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss
        
        # Final layer norm
        x = self.ln_f(x)
        
        # LM head to get logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss if targets provided
        lm_loss = None
        if targets is not None:
            # Reshape for cross-entropy: (batch * seq_len, vocab_size) and (batch * seq_len,)
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            lm_loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, lm_loss, total_aux_loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            eos_token_id: End-of-sequence token ID (stops generation if encountered)
            
        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop input to max_seq_len if needed
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _, _ = self(input_ids_cond)
            
            # Get logits for last token
            logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token generated (for all sequences in batch)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            n_params -= self.pos_embeddings.weight.numel()
        
        return n_params


def create_model(config) -> TransformerLM:
    """Create a TransformerLM model from a ModelConfig.
    
    Args:
        config: ModelConfig instance
        
    Returns:
        TransformerLM model
    """
    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        attention_type=config.attention_type,
        n_kv_heads=config.n_kv_heads,
        ffn_type=config.ffn_type,
        num_experts=config.num_experts,
        top_k=config.top_k,
    )
    
    return model
