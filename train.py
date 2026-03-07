#!/usr/bin/env python3
"""Main training script for Transformer language model."""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config_from_yaml, merge_config_overrides, parse_override_string
from src.utils.misc import set_seed, count_parameters
from src.data.tokenizer import load_tokenizer
from src.data.lm_dataset import CausalLMDataset, collate_fn
from src.models.gpt import create_model
from src.train.trainer import Trainer
from src.eval.sample import generate_samples_for_eval, print_generation_samples


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Transformer language model')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--set',
        type=str,
        action='append',
        help='Override config values (e.g., --set model.n_layers=4)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation, no training'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    print(f"📋 Loading configuration from {args.config}...")
    config = load_config_from_yaml(args.config)
    
    # Apply overrides
    if args.set:
        overrides = {}
        for override_str in args.set:
            key, value = parse_override_string(override_str)
            overrides[key] = value
        config = merge_config_overrides(config, overrides)
        print(f"   Applied {len(overrides)} config override(s)")
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Print configuration
    print(f"\n{'='*70}")
    print(f"{'Model Configuration':^70}")
    print('='*70)
    print(f"  Architecture: {config.model.attention_type.upper()} attention + {config.model.ffn_type.upper()} FFN")
    print(f"  Layers: {config.model.n_layers}")
    print(f"  Model dim: {config.model.d_model}")
    print(f"  Heads: {config.model.n_heads} (KV heads: {config.model.n_kv_heads})")
    print(f"  FFN dim: {config.model.d_ff}")
    print(f"  Vocab size: {config.model.vocab_size}")
    print(f"  Max seq len: {config.model.max_seq_len}")
    if config.model.ffn_type == 'moe':
        print(f"  MoE: {config.model.num_experts} experts, top-{config.model.top_k}")
    print('='*70)
    
    # Load tokenizer
    print(f"\n📖 Loading tokenizer from {config.tokenizer_path}...")
    tokenizer = load_tokenizer(config.tokenizer_path)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Verify vocab size matches
    if config.model.vocab_size != tokenizer.vocab_size:
        print(f"⚠️  Warning: Config vocab_size ({config.model.vocab_size}) != "
              f"tokenizer vocab_size ({tokenizer.vocab_size})")
        print(f"   Updating config to match tokenizer")
        config.model.vocab_size = tokenizer.vocab_size
    
    # Load datasets
    print(f"\n📚 Loading datasets...")
    train_token_ids = torch.load(config.training.train_data_path)
    val_token_ids = torch.load(config.training.val_data_path)
    
    train_dataset = CausalLMDataset(train_token_ids, config.model.max_seq_len)
    val_dataset = CausalLMDataset(val_token_ids, config.model.max_seq_len)
    
    print(f"   Train: {len(train_dataset):,} examples ({len(train_token_ids):,} tokens)")
    print(f"   Val:   {len(val_dataset):,} examples ({len(val_token_ids):,} tokens)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for now
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = create_model(config.model)
    
    total_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        moe_aux_loss_weight=config.model.moe_aux_loss_weight
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Eval-only mode
    if args.eval_only:
        print(f"\n📊 Running evaluation...")
        val_metrics = trainer.validate()
        print(f"\n{'='*70}")
        print(f"{'Validation Results':^70}")
        print('='*70)
        print(f"  Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Perplexity: {val_metrics['val_ppl']:.2f}")
        print('='*70)
        
        # Generate samples
        samples = generate_samples_for_eval(
            model, tokenizer,
            num_samples=config.training.num_gen_samples,
            max_tokens=config.training.gen_max_tokens,
            temperature=config.training.gen_temperature,
            top_p=config.training.gen_top_p,
            device=trainer.device
        )
        print_generation_samples(samples)
        
        return
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"{'Starting Training':^70}")
    print('='*70)
    print(f"  Max steps: {config.training.max_steps:,}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.grad_accum_steps}")
    print(f"  Effective batch size: {config.training.batch_size * config.training.grad_accum_steps}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Warmup steps: {config.training.warmup_steps:,}")
    print(f"  Device: {trainer.device}")
    print(f"  Mixed precision: {trainer.use_amp}")
    print('='*70 + '\n')
    
    # Progress bar
    pbar = tqdm(total=config.training.max_steps, initial=trainer.global_step)
    
    try:
        while trainer.global_step < config.training.max_steps:
            # Training step
            metrics = trainer.train_step()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{metrics['lr']:.2e}",
                'gnorm': f"{metrics['grad_norm']:.2f}"
            })
            
            # Log metrics
            if trainer.global_step % config.training.log_every == 0:
                trainer.log_metrics(metrics)
            
            # Validation
            if trainer.global_step % config.training.eval_every == 0:
                print(f"\n📊 Validation at step {trainer.global_step}...")
                val_metrics = trainer.validate()
                
                # Combine train and val metrics
                all_metrics = {**metrics, **val_metrics}
                trainer.log_metrics(all_metrics)
                
                print(f"   Val Loss: {val_metrics['val_loss']:.4f} | PPL: {val_metrics['val_ppl']:.2f}")
                
                # Generate samples
                print(f"   Generating samples...")
                samples = generate_samples_for_eval(
                    model, tokenizer,
                    num_samples=config.training.num_gen_samples,
                    max_tokens=config.training.gen_max_tokens,
                    temperature=config.training.gen_temperature,
                    top_p=config.training.gen_top_p,
                    device=trainer.device
                )
                print_generation_samples(samples, title=f"Samples @ Step {trainer.global_step}")
                
                # Save best model
                if val_metrics['val_loss'] < trainer.best_val_loss:
                    trainer.best_val_loss = val_metrics['val_loss']
                    checkpoint_path = trainer.save_checkpoint('best.pt')
                    print(f"   💾 Saved best model: {checkpoint_path}")
            
            # Save checkpoint
            if trainer.global_step % config.training.save_every == 0:
                checkpoint_path = trainer.save_checkpoint()
                print(f"\n💾 Saved checkpoint: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        checkpoint_path = trainer.save_checkpoint('interrupted.pt')
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    finally:
        pbar.close()
    
    print(f"\n{'='*70}")
    print(f"{'Training Complete':^70}")
    print('='*70)
    print(f"  Final step: {trainer.global_step}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
