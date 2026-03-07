#!/usr/bin/env python3
"""Script to train a SentencePiece tokenizer from raw text data."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import train_tokenizer, load_tokenizer, clean_text


def main():
    parser = argparse.ArgumentParser(description='Train a SentencePiece tokenizer')
    parser.add_argument(
        '--input',
        type=str,
        default='data/data-model.csv',
        help='Input text file for training (default: data/raw.txt)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='artifacts/tokenizer',
        help='Output prefix for model files (default: artifacts/tokenizer)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=6370,
        help='Vocabulary size (default: 6370)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='bpe',
        choices=['bpe', 'unigram', 'char', 'word'],
        help='Tokenizer model type (default: bpe)'
    )
    parser.add_argument(
        '--clean-input',
        action='store_true',
        help='Clean input text before training'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the trained tokenizer with sample text'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        print(f"\nPlease create the file or provide a different path using --input")
        sys.exit(1)
    
    # Clean input if requested
    if args.clean_input:
        print(f"🧹 Cleaning input text from {args.input}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned_text = clean_text(text, remove_empty_lines=True)
        
        # Save cleaned text to a temporary file
        cleaned_path = input_path.parent / f"{input_path.stem}_cleaned.txt"
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"✓ Cleaned text saved to {cleaned_path}")
        input_file = str(cleaned_path)
    else:
        input_file = args.input
    
    # Train the tokenizer
    print(f"\n🔧 Training SentencePiece tokenizer...")
    print(f"   Input: {input_file}")
    print(f"   Output: {args.output_prefix}.model")
    print(f"   Vocab size: {args.vocab_size}")
    print(f"   Model type: {args.model_type}")
    print()
    
    train_tokenizer(
        input_file=input_file,
        output_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type
    )
    
    # Test the tokenizer if requested
    if args.test:
        print(f"\n🧪 Testing tokenizer...")
        tokenizer = load_tokenizer(f"{args.output_prefix}.model")
        
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
        ]
        
        print(f"\nTokenizer: {tokenizer}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Special tokens: PAD={tokenizer.pad_id}, UNK={tokenizer.unk_id}, "
              f"BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
        
        for text in test_texts:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            
            print(f"\n  Text: {text}")
            print(f"  IDs: {ids[:20]}{'...' if len(ids) > 20 else ''} (len={len(ids)})")
            print(f"  Decoded: {decoded}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
