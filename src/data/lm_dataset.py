"""Dataset for causal language modeling."""

import torch
from torch.utils.data import Dataset
from typing import Tuple


class CausalLMDataset(Dataset):
    """Dataset for causal language modeling (next token prediction).
    
    Takes a 1D tensor of token IDs and creates overlapping windows where
    the target is the input shifted by one position.
    
    Args:
        token_ids: 1D tensor of token IDs
        seq_len: Sequence length for each training example
        stride: Stride for creating windows (default: seq_len for non-overlapping)
    """
    
    def __init__(self, token_ids: torch.Tensor, seq_len: int, stride: int = None):
        if token_ids.dim() != 1:
            raise ValueError(f"token_ids must be a 1D tensor, got shape {token_ids.shape}")
        
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        
        # Calculate number of examples
        # We need seq_len + 1 tokens to create (input, target) pair
        total_tokens = len(token_ids)
        if total_tokens <= seq_len:
            raise ValueError(
                f"Dataset has {total_tokens} tokens but needs at least {seq_len + 1} "
                f"tokens for seq_len={seq_len}"
            )
        
        # Number of valid starting positions
        self.num_examples = max(1, (total_tokens - seq_len - 1) // self.stride + 1)
    
    def __len__(self) -> int:
        return self.num_examples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tuple of (input_ids, target_ids) where target_ids = input_ids shifted by 1
        """
        # Calculate starting position
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len + 1
        
        # Extract tokens
        tokens = self.token_ids[start_idx:end_idx]
        
        # Split into input and target
        input_ids = tokens[:-1]  # All except last
        target_ids = tokens[1:]  # All except first
        
        return input_ids, target_ids


def collate_fn(batch):
    """Collate function for DataLoader.
    
    Args:
        batch: List of (input_ids, target_ids) tuples
        
    Returns:
        Tuple of (input_batch, target_batch) as tensors
    """
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return inputs, targets
