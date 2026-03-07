#!/usr/bin/env python3
"""Script to tokenize raw text and build token ID cache for training."""

import argparse
from pathlib import Path
import sys
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Tokenize raw text and create train/val token caches'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/data-model.csv',
        help='Input text file (default: data/raw.txt)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='artifacts/tokenizer.model',
        help='Path to trained tokenizer model (default: artifacts/tokenizer.model)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts',
        help='Output directory for token caches (default: artifacts)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Fraction of data to use for validation (default: 0.1)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Maximum number of tokens to process (default: None = all)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check if tokenizer exists
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"❌ Error: Tokenizer not found: {args.tokenizer}")
        print(f"   Please train a tokenizer first using scripts/train_tokenizer.py")
        sys.exit(1)
    
    # Load tokenizer
    print(f"📖 Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Read input text
    print(f"\n📄 Reading input text from {args.input}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Text size: {len(text):,} characters")
    
    # Tokenize
    print(f"\n🔤 Tokenizing text...")
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False, clean=True)
    total_tokens = len(token_ids)
    
    print(f"   Total tokens: {total_tokens:,}")
    
    # Truncate if requested
    if args.max_tokens is not None and total_tokens > args.max_tokens:
        print(f"   Truncating to {args.max_tokens:,} tokens")
        token_ids = token_ids[:args.max_tokens]
        total_tokens = len(token_ids)
    
    # Convert to tensor
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    
    # Split into train and validation
    val_size = int(total_tokens * args.val_split)
    train_size = total_tokens - val_size
    
    train_ids = token_tensor[:train_size]
    val_ids = token_tensor[train_size:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_ids):,} tokens ({len(train_ids)/total_tokens*100:.1f}%)")
    print(f"   Val:   {len(val_ids):,} tokens ({len(val_ids)/total_tokens*100:.1f}%)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save token caches
    train_path = output_dir / 'train_ids.pt'
    val_path = output_dir / 'val_ids.pt'
    
    print(f"\n💾 Saving token caches...")
    torch.save(train_ids, train_path)
    torch.save(val_ids, val_path)
    
    print(f"   ✓ Train: {train_path}")
    print(f"   ✓ Val:   {val_path}")
    
    # Print some statistics
    print(f"\n📈 Token statistics:")
    print(f"   Unique tokens (train): {len(torch.unique(train_ids)):,}")
    print(f"   Unique tokens (val):   {len(torch.unique(val_ids)):,}")
    print(f"   Vocab coverage (train): {len(torch.unique(train_ids))/tokenizer.vocab_size*100:.2f}%")
    
    # Show sample
    print(f"\n🔍 Sample from training data (first 100 tokens):")
    sample_ids = train_ids[:100].tolist()
    sample_text = tokenizer.decode(sample_ids)
    print(f"   {sample_text[:200]}{'...' if len(sample_text) > 200 else ''}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
