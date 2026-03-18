#!/usr/bin/env python3
"""Inference script for text generation from a trained Transformer LM checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config_from_yaml, merge_config_overrides, parse_override_string
from src.data.tokenizer import load_tokenizer
from src.models.gpt import create_model
from src.eval.sample import generate


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (same one used for training)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. 0 = greedy, higher = more random (default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold (default: 0.95)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--set",
        type=str,
        action="append",
        help="Override config values (e.g., --set model.n_layers=4)",
    )

    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    """Load model from config and checkpoint.

    Args:
        config: Config object with model architecture settings.
        checkpoint_path: Path to the saved checkpoint.
        device: torch.device to load onto.

    Returns:
        Loaded TransformerLM model in eval mode.
    """
    model = create_model(config.model)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    step = checkpoint.get("global_step", "?")
    print(f"   Loaded checkpoint from step {step}")

    return model


def interactive_mode(model, tokenizer, device, max_tokens, temperature, top_p):
    """Run interactive generation loop.

    Args:
        model: Loaded TransformerLM model.
        tokenizer: TokenizerWrapper instance.
        device: torch.device for inference.
        max_tokens: Default max tokens to generate.
        temperature: Default sampling temperature.
        top_p: Default nucleus sampling threshold.
    """
    print("\n" + "=" * 70)
    print("Interactive Generation Mode".center(70))
    print("=" * 70)
    print("  Type a prompt and press Enter to generate.")
    print("  Commands:")
    print("    /quit          - Exit")
    print("    /temp <value>  - Set temperature (current: {:.2f})".format(temperature))
    print("    /topp <value>  - Set top-p (current: {:.2f})".format(top_p))
    print("    /max <value>   - Set max tokens (current: {})".format(max_tokens))
    print("=" * 70 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.startswith("/"):
            parts = prompt.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("Exiting.")
                break
            elif cmd == "/temp" and len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"  Temperature set to {temperature:.2f}")
                except ValueError:
                    print("  Invalid value. Usage: /temp 0.8")
            elif cmd == "/topp" and len(parts) == 2:
                try:
                    top_p = float(parts[1])
                    print(f"  Top-p set to {top_p:.2f}")
                except ValueError:
                    print("  Invalid value. Usage: /topp 0.95")
            elif cmd == "/max" and len(parts) == 2:
                try:
                    max_tokens = int(parts[1])
                    print(f"  Max tokens set to {max_tokens}")
                except ValueError:
                    print("  Invalid value. Usage: /max 200")
            else:
                print("  Unknown command. Available: /quit, /temp, /topp, /max")
            continue

        # Generate
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

        print(f"\n{output}\n")


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    print(f"Loading configuration from {args.config}...")
    config = load_config_from_yaml(args.config)

    if args.set:
        overrides = {}
        for override_str in args.set:
            key, value = parse_override_string(override_str)
            overrides[key] = value
        config = merge_config_overrides(config, overrides)

    # Load tokenizer
    print(f"Loading tokenizer from {config.tokenizer_path}...")
    tokenizer = load_tokenizer(config.tokenizer_path)

    # Match vocab size
    if config.model.vocab_size != tokenizer.vocab_size:
        config.model.vocab_size = tokenizer.vocab_size

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(config, args.checkpoint, device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")
    print(f"   Device: {device}")

    if args.prompt is not None:
        # Single-shot generation
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(f"\n{output}")
    else:
        # Interactive mode
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
