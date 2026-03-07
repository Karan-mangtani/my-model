# Transformer Training Framework

A modular PyTorch framework for training decoder-only Transformer language models from scratch. Designed for learning and experimentation with modern LLM architectures, including different attention mechanisms (MHA/MQA/GQA) and feed-forward variants (Dense/MoE).

## Features

- **Multiple Attention Mechanisms**
  - Multi-Head Attention (MHA) - standard attention
  - Multi-Query Attention (MQA) - shared K/V heads for efficient inference
  - Grouped-Query Attention (GQA) - middle ground between MHA and MQA

- **Feed-Forward Network Variants**
  - Dense FFN - standard feedforward layers
  - Mixture of Experts (MoE) - sparse routing with load balancing

- **Training Optimizations**
  - Mixed precision training (AMP)
  - Gradient accumulation for large effective batch sizes
  - Gradient clipping
  - Cosine learning rate schedule with linear warmup
  - Automatic checkpointing

- **Memory Efficient for Small GPUs**
  - Designed to run on 4GB GPUs
  - Uses PyTorch 2.0's optimized `scaled_dot_product_attention`
  - Efficient data loading with memory-mapped tensors

## Project Structure

```
my-model/
├── configs/                    # YAML configuration files
│   ├── base_dense_mha.yaml     # Baseline: MHA + Dense FFN
│   ├── dense_mqa.yaml          # MQA + Dense FFN
│   ├── dense_gqa.yaml          # GQA + Dense FFN
│   └── moe_mha.yaml            # MHA + MoE FFN
├── src/
│   ├── data/                   # Data loading and tokenization
│   │   ├── tokenizer.py        # SentencePiece tokenizer wrapper
│   │   └── lm_dataset.py       # Causal language modeling dataset
│   ├── models/                 # Model architectures
│   │   ├── attention.py        # Attention variants (MHA/MQA/GQA)
│   │   ├── ffn.py              # FFN variants (Dense/MoE)
│   │   ├── transformer.py      # Transformer block
│   │   └── gpt.py              # Full GPT-style model
│   ├── train/                  # Training infrastructure
│   │   └── trainer.py          # Trainer with AMP, grad accumulation, etc.
│   ├── eval/                   # Evaluation and generation
│   │   ├── evaluate.py         # Compute validation metrics
│   │   └── sample.py           # Text generation utilities
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration management
│       └── misc.py             # Helper functions
├── scripts/                    # Standalone scripts
│   ├── train_tokenizer.py      # Train SentencePiece tokenizer
│   └── build_token_cache.py    # Tokenize and cache training data
├── data/                       # Raw text data (gitignored)
├── artifacts/                  # Tokenizers, checkpoints, caches (gitignored)
├── train.py                    # Main training entrypoint
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Installation

1. Clone or download this repository:
```bash
cd my-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Training Data

Place your raw text data in `data/raw.txt`. For example:

```bash
# Download a sample dataset (e.g., TinyStories, WikiText, or your own text)
# For this example, we'll assume you have a text file ready
echo "Your training text goes here..." > data/raw.txt
```

### 2. Train Tokenizer

Train a SentencePiece BPE tokenizer on your data:

```bash
python scripts/train_tokenizer.py \
    --input data/raw.txt \
    --output-prefix artifacts/tokenizer \
    --vocab-size 8192 \
    --test
```

This creates:
- `artifacts/tokenizer.model` - tokenizer model
- `artifacts/tokenizer.vocab` - vocabulary file

### 3. Build Token Cache

Tokenize the raw text and create train/val splits:

```bash
python scripts/build_token_cache.py \
    --input data/raw.txt \
    --tokenizer artifacts/tokenizer.model \
    --output-dir artifacts \
    --val-split 0.1
```

This creates:
- `artifacts/train_ids.pt` - training token IDs
- `artifacts/val_ids.pt` - validation token IDs

### 4. Train Model

Train with one of the provided configurations:

```bash
# Base configuration: MHA + Dense FFN
python train.py --config configs/base_dense_mha.yaml

# Or try MQA for faster inference
python train.py --config configs/dense_mqa.yaml

# Or experiment with MoE
python train.py --config configs/moe_mha.yaml
```

### 5. Monitor Training

Training logs are saved to `artifacts/checkpoints/training_log.csv`. Checkpoints are saved every 2000 steps to `artifacts/checkpoints/`.

The training script will:
- Show loss, perplexity, and learning rate
- Generate sample text periodically
- Save checkpoints automatically
- Track best validation loss

## Configuration

All training configurations are in YAML files under `configs/`. You can override any setting from the command line:

```bash
# Override specific settings
python train.py --config configs/base_dense_mha.yaml \
    --set model.n_layers=4 \
    --set training.learning_rate=1e-4

# Train a smaller model
python train.py --config configs/base_dense_mha.yaml \
    --set model.d_model=256 \
    --set model.n_layers=4 \
    --set model.n_heads=4
```

### Key Configuration Parameters

**Model Architecture:**
- `d_model`: Model dimension (default: 384)
- `n_layers`: Number of transformer layers (default: 6)
- `n_heads`: Number of attention heads (default: 6)
- `d_ff`: Feed-forward dimension (default: 1536, i.e., 4×d_model)
- `max_seq_len`: Maximum sequence length (default: 256)

**Attention Type:**
- `attention_type`: `mha`, `mqa`, or `gqa`
- `n_kv_heads`: Number of K/V heads (1 for MQA, <n_heads for GQA)

**FFN Type:**
- `ffn_type`: `dense` or `moe`
- `num_experts`: Number of experts for MoE (default: 8)
- `top_k`: Experts per token for MoE (default: 2)

**Training:**
- `batch_size`: Batch size per GPU (default: 1)
- `grad_accum_steps`: Gradient accumulation steps (default: 32)
- `learning_rate`: Peak learning rate (default: 3e-4)
- `max_steps`: Total training steps (default: 50000)
- `warmup_steps`: LR warmup steps (default: 2000)

## Attention Mechanisms Explained

### Multi-Head Attention (MHA)
Standard attention where each query head has its own key and value heads. Most expressive but uses most memory.

**Use case:** Best quality, baseline for comparison

### Multi-Query Attention (MQA)
All query heads share a single key and value head. Significantly reduces KV cache size during inference.

**Use case:** Fastest inference, ~4x smaller KV cache

### Grouped-Query Attention (GQA)
Query heads are grouped, with each group sharing K/V heads. Balances quality and efficiency.

**Use case:** Production models (used in LLaMA 2 70B), good quality-speed tradeoff

## Feed-Forward Network Variants

### Dense FFN
Standard two-layer MLP with GELU activation:
```
x → Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
```

**Use case:** Simpler, fewer parameters, easier to train

### Mixture of Experts (MoE)
Multiple expert networks with learned routing. Each token is routed to top-k experts:
```
router(x) → top_k_experts → weighted_combine(expert_outputs)
```

Includes load balancing loss to encourage uniform expert usage.

**Use case:** More capacity with fewer active parameters, can scale to very large models

## GPU Memory Requirements

**Base configuration** (d_model=384, n_layers=6, seq_len=256):
- Model: ~25M parameters ≈ 100MB
- Activations (batch_size=1): ~50-100MB
- Optimizer states: ~300MB
- **Total: ~500MB** (fits comfortably in 4GB)

For larger models or longer sequences, increase `grad_accum_steps` instead of `batch_size`.

## Advanced Usage

### Resume Training

```bash
python train.py --config configs/base_dense_mha.yaml \
    --resume artifacts/checkpoints/latest.pt
```

### Evaluation Only

```bash
python train.py --config configs/base_dense_mha.yaml \
    --resume artifacts/checkpoints/best.pt \
    --eval-only
```

### Custom Configurations

Create your own config file:

```yaml
# configs/my_config.yaml
tokenizer_path: artifacts/tokenizer.model

model:
  vocab_size: 8192
  d_model: 512
  n_layers: 8
  n_heads: 8
  d_ff: 2048
  attention_type: gqa
  n_kv_heads: 2
  ffn_type: moe
  num_experts: 16
  top_k: 2

training:
  batch_size: 1
  grad_accum_steps: 64
  learning_rate: 3.0e-4
  max_steps: 100000
  # ... other settings
```

## Tips for Small GPUs

1. **Reduce sequence length:** `--set model.max_seq_len=128`
2. **Increase gradient accumulation:** `--set training.grad_accum_steps=64`
3. **Use smaller model:** `--set model.d_model=256 --set model.n_layers=4`
4. **Disable mixed precision if issues:** `--set training.mixed_precision=false`

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` (already 1 by default)
- Increase `grad_accum_steps` (maintains effective batch size)
- Reduce `max_seq_len`
- Reduce model size (`d_model`, `n_layers`)

**Training is slow:**
- Ensure CUDA is available: check `device: cuda` in config
- Consider using sequence length curriculum (start small, increase over time)

**Loss not decreasing:**
- Check data quality and tokenization
- Verify effective batch size is reasonable (batch_size × grad_accum_steps ≥ 16)
- Try different learning rates
- Ensure warmup_steps is appropriate

**Tokenizer not found:**
- Run `scripts/train_tokenizer.py` first
- Check `tokenizer_path` in config matches actual file location

## Architecture Details

**Pre-norm Transformer:**
- Uses pre-normalization (LayerNorm before attention/FFN)
- More stable training than post-norm
- Allows higher learning rates

**Initialization:**
- Token embeddings: N(0, 0.02)
- Residual projections: Scaled by 1/√n_layers
- Supports deep models without careful tuning

**Weight Tying:**
- Token embeddings and LM head share weights
- Reduces parameters and improves perplexity

## Citation

This is an educational implementation. If you use it for research, please cite the relevant papers:
- Attention: Vaswani et al. (2017) - "Attention Is All You Need"
- MQA: Shazeer (2019) - "Fast Transformer Decoding"
- GQA: Ainslie et al. (2023) - "GQA: Training Generalized Multi-Query Transformer"
- MoE: Shazeer et al. (2017) - "Outrageously Large Neural Networks"

## License

MIT License - feel free to use for learning and experimentation.
"# my-model" 
