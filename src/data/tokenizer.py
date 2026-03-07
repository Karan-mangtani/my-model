"""Tokenizer module using SentencePiece for BPE tokenization."""

import sentencepiece as spm
from pathlib import Path
from typing import List, Union, Optional
import re


def clean_text(text: str, remove_empty_lines: bool = True) -> str:
    """Clean text by normalizing whitespace and optionally removing empty lines.
    
    Args:
        text: Input text to clean
        remove_empty_lines: Whether to remove empty lines
        
    Returns:
        Cleaned text
    """
    # Normalize unicode
    text = text.strip()
    
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    if remove_empty_lines:
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
    else:
        # Just clean up excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def train_tokenizer(
    input_file: str,
    output_prefix: str,
    vocab_size: int = 8192,
    model_type: str = 'bpe',
    character_coverage: float = 0.9995,
    max_sentence_length: int = 16384,
) -> None:
    """Train a SentencePiece tokenizer.
    
    Args:
        input_file: Path to input text file for training
        output_prefix: Prefix for output model files (.model and .vocab)
        vocab_size: Target vocabulary size
        model_type: Type of tokenizer ('bpe', 'unigram', 'char', or 'word')
        character_coverage: Character coverage for vocabulary
        max_sentence_length: Maximum sentence length
    """
    # Ensure output directory exists
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define special tokens
    user_defined_symbols = []
    
    # Train the tokenizer
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=output_prefix,
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        max_sentence_length=max_sentence_length,
        pad_id=0,  # <pad>
        unk_id=1,  # <unk>
        bos_id=2,  # <bos>
        eos_id=3,  # <eos>
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<bos>',
        eos_piece='<eos>',
        user_defined_symbols=user_defined_symbols,
        num_threads=8,
        split_digits=True,  # Split digits into individual tokens
        byte_fallback=True,  # Use byte fallback for unknown characters
    )
    
    print(f"✓ Tokenizer trained and saved to {output_prefix}.model")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model type: {model_type}")


class TokenizerWrapper:
    """Wrapper for SentencePiece tokenizer with convenient methods."""
    
    def __init__(self, model_path: str):
        """Initialize the tokenizer wrapper.
        
        Args:
            model_path: Path to the trained SentencePiece model
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
        
        # Cache special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.vocab_size()
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        clean: bool = True
    ) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            clean: Whether to clean the text before encoding
            
        Returns:
            List of token IDs
        """
        if clean:
            text = clean_text(text)
        
        # Encode using SentencePiece
        ids = self.sp.encode(text, out_type=int)
        
        # Add special tokens
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """Decode token IDs to text.
        
        Args:
            ids: Token IDs or list of token ID sequences
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text or list of decoded texts
        """
        # Check if it's a batch of sequences
        if ids and isinstance(ids[0], list):
            return [self.decode(seq, skip_special_tokens) for seq in ids]
        
        # Filter special tokens if requested
        if skip_special_tokens:
            ids = [id for id in ids if id not in {self.pad_id, self.bos_id, self.eos_id, self.unk_id}]
        
        # Decode using SentencePiece
        text = self.sp.decode(ids)
        return text
    
    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        clean: bool = True
    ) -> List[List[int]]:
        """Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            clean: Whether to clean the texts before encoding
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_bos, add_eos, clean) for text in texts]
    
    def __repr__(self) -> str:
        return f"TokenizerWrapper(model_path='{self.model_path}', vocab_size={self.vocab_size})"


def load_tokenizer(model_path: str) -> TokenizerWrapper:
    """Load a trained tokenizer.
    
    Args:
        model_path: Path to the trained SentencePiece model
        
    Returns:
        TokenizerWrapper instance
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Tokenizer model not found at {model_path}")
    
    return TokenizerWrapper(model_path)
