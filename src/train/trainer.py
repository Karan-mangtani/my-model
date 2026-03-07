"""Trainer for Transformer language models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import csv


class Trainer:
    """Trainer for Transformer language models.
    
    Handles:
        - Mixed precision training (AMP)
        - Gradient accumulation
        - Gradient clipping
        - Learning rate scheduling (cosine with warmup)
        - Checkpointing
        - Logging
    
    Args:
        model: TransformerLM model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: TrainingConfig instance
        moe_aux_loss_weight: Weight for MoE auxiliary loss
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        moe_aux_loss_weight: float = 0.01
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.moe_aux_loss_weight = moe_aux_loss_weight
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer - exclude LayerNorm and bias from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'ln' in name or 'bias' in name or 'norm' in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=config.learning_rate)
        
        # Mixed precision
        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV logging
        self.log_file = self.checkpoint_dir / 'training_log.csv'
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize CSV log file."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'epoch', 'train_loss', 'train_lm_loss', 'train_aux_loss',
                    'val_loss', 'val_ppl', 'lr', 'grad_norm'
                ])
    
    def _get_lr(self) -> float:
        """Get current learning rate with warmup and cosine decay."""
        step = self.global_step
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps
        lr = self.config.learning_rate
        
        if step < warmup_steps:
            # Linear warmup
            return lr * (step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_step(self) -> Dict[str, float]:
        """Execute one training step with gradient accumulation.
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Update learning rate
        current_lr = self._get_lr()
        self._set_lr(current_lr)
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_aux_loss = 0.0
        
        # Gradient accumulation
        self.optimizer.zero_grad()
        
        for micro_step in range(self.config.grad_accum_steps):
            # Get batch
            try:
                input_ids, targets = next(self.train_iter)
            except (StopIteration, AttributeError):
                self.train_iter = iter(self.train_loader)
                input_ids, targets = next(self.train_iter)
            
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits, lm_loss, aux_loss = self.model(input_ids, targets)
                
                # Combined loss
                loss = lm_loss + self.moe_aux_loss_weight * aux_loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate metrics
            total_loss += loss.item() * self.config.grad_accum_steps
            total_lm_loss += lm_loss.item()
            total_aux_loss += aux_loss.item()
        
        # Gradient clipping (unscale first if using AMP)
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        ).item()
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.global_step += 1
        
        # Average metrics over accumulation steps
        metrics = {
            'loss': total_loss / self.config.grad_accum_steps,
            'lm_loss': total_lm_loss / self.config.grad_accum_steps,
            'aux_loss': total_aux_loss / self.config.grad_accum_steps,
            'lr': current_lr,
            'grad_norm': grad_norm
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits, lm_loss, aux_loss = self.model(input_ids, targets)
                loss = lm_loss + self.moe_aux_loss_weight * aux_loss
            
            batch_size, seq_len = input_ids.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_ppl': perplexity
        }
    
    def save_checkpoint(self, filename: str = None):
        """Save checkpoint.
        
        Args:
            filename: Checkpoint filename (default: step_{global_step}.pt)
        """
        if filename is None:
            filename = f'step_{self.global_step}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': {
                'learning_rate': self.config.learning_rate,
                'warmup_steps': self.config.warmup_steps,
                'max_steps': self.config.max_steps,
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler is not None and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Resuming from step {self.global_step}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to CSV file.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.global_step,
                self.epoch,
                metrics.get('loss', ''),
                metrics.get('lm_loss', ''),
                metrics.get('aux_loss', ''),
                metrics.get('val_loss', ''),
                metrics.get('val_ppl', ''),
                metrics.get('lr', ''),
                metrics.get('grad_norm', '')
            ])
