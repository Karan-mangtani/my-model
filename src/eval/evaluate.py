"""Evaluation utilities for language models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import math
from typing import Dict


@torch.no_grad()
def compute_validation_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    moe_aux_loss_weight: float = 0.01
) -> Dict[str, float]:
    """Compute validation loss and perplexity.
    
    Args:
        model: TransformerLM model
        val_loader: Validation data loader
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision
        moe_aux_loss_weight: Weight for MoE auxiliary loss
        
    Returns:
        Dictionary with 'loss' and 'perplexity' keys
    """
    model.eval()
    
    total_loss = 0.0
    total_lm_loss = 0.0
    total_aux_loss = 0.0
    total_tokens = 0
    
    for input_ids, targets in val_loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        with autocast(enabled=use_amp):
            logits, lm_loss, aux_loss = model(input_ids, targets)
            loss = lm_loss + moe_aux_loss_weight * aux_loss
        
        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        
        total_loss += loss.item() * num_tokens
        total_lm_loss += lm_loss.item() * num_tokens
        total_aux_loss += aux_loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    avg_lm_loss = total_lm_loss / total_tokens
    avg_aux_loss = total_aux_loss / total_tokens
    perplexity = math.exp(avg_lm_loss)  # Use LM loss for perplexity
    
    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'aux_loss': avg_aux_loss,
        'perplexity': perplexity
    }
