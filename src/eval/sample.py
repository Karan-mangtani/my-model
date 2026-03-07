"""Text generation and sampling utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, List[int]],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device = None
) -> str:
    """Generate text from a prompt.
    
    Args:
        model: TransformerLM model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt (string or token IDs)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold (1.0 = no filtering)
        device: Device to run on
        
    Returns:
        Generated text
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Encode prompt if string
    if isinstance(prompt, str):
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    else:
        input_ids = prompt
    
    # Convert to tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate using model's generate method
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_id
    )
    
    # Decode
    output_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return generated_text


@torch.no_grad()
def generate_batch(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device = None
) -> List[str]:
    """Generate text from multiple prompts.
    
    Args:
        model: TransformerLM model
        tokenizer: Tokenizer with encode/decode methods
        prompts: List of input prompts
        max_tokens: Maximum number of tokens to generate per prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Device to run on
        
    Returns:
        List of generated texts
    """
    return [
        generate(model, tokenizer, prompt, max_tokens, temperature, top_p, device)
        for prompt in prompts
    ]


def generate_samples_for_eval(
    model: nn.Module,
    tokenizer,
    num_samples: int = 3,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: torch.device = None
) -> List[tuple]:
    """Generate sample texts for evaluation during training.
    
    Uses fixed prompts to monitor generation quality over time.
    
    Args:
        model: TransformerLM model
        tokenizer: Tokenizer
        num_samples: Number of samples to generate
        max_tokens: Maximum tokens per sample
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Device to run on
        
    Returns:
        List of (prompt, generated_text) tuples
    """
    # Fixed prompts for consistent evaluation
    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In a world where",
        "It was a dark and stormy night",
        "The secret to happiness",
    ]
    
    # Select subset
    prompts = prompts[:num_samples]
    
    results = []
    for prompt in prompts:
        generated = generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device
        )
        results.append((prompt, generated))
    
    return results


def print_generation_samples(samples: List[tuple], title: str = "Generation Samples"):
    """Pretty print generation samples.
    
    Args:
        samples: List of (prompt, generated_text) tuples
        title: Title to print
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print('='*70)
    
    for i, (prompt, generated) in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {generated}")
    
    print('='*70)
