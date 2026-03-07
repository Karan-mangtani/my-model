"""Configuration management for the Transformer training system."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the Transformer model architecture."""
    
    # Architecture basics
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536  # Typically 4 * d_model
    dropout: float = 0.1
    max_seq_len: int = 256
    
    # Attention configuration
    attention_type: str = "mha"  # Options: "mha", "mqa", "gqa"
    n_kv_heads: Optional[int] = None  # None = MHA, 1 = MQA, other = GQA
    
    # FFN configuration
    ffn_type: str = "dense"  # Options: "dense", "moe"
    num_experts: int = 8  # For MoE
    top_k: int = 2  # For MoE routing
    moe_aux_loss_weight: float = 0.01  # For MoE load balancing
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set n_kv_heads based on attention_type if not explicitly set
        if self.attention_type == "mha" and self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        elif self.attention_type == "mqa" and self.n_kv_heads is None:
            self.n_kv_heads = 1
        elif self.attention_type == "gqa" and self.n_kv_heads is None:
            # Default to n_heads // 2 for GQA
            self.n_kv_heads = max(1, self.n_heads // 2)
        
        # Validate attention configuration
        assert self.attention_type in ["mha", "mqa", "gqa"], \
            f"Invalid attention_type: {self.attention_type}"
        assert self.n_kv_heads <= self.n_heads, \
            f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        
        # Validate FFN configuration
        assert self.ffn_type in ["dense", "moe"], \
            f"Invalid ffn_type: {self.ffn_type}"
        if self.ffn_type == "moe":
            assert self.num_experts > 0, "num_experts must be positive"
            assert 0 < self.top_k <= self.num_experts, \
                f"top_k ({self.top_k}) must be between 1 and num_experts ({self.num_experts})"


@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    
    # Batch and optimization
    batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 2000
    max_steps: int = 50000
    lr_schedule: str = "cosine"  # Options: "cosine", "linear", "constant"
    
    # Evaluation and checkpointing
    eval_every: int = 500
    save_every: int = 2000
    log_every: int = 10
    
    # Generation settings for evaluation
    gen_max_tokens: int = 100
    gen_temperature: float = 0.8
    gen_top_p: float = 0.95
    num_gen_samples: int = 2
    
    # Data paths
    train_data_path: str = "artifacts/train_ids.pt"
    val_data_path: str = "artifacts/val_ids.pt"
    checkpoint_dir: str = "artifacts/checkpoints"
    
    # Hardware
    device: str = "cuda"  # "cuda" or "cpu"
    mixed_precision: bool = True  # Use AMP
    
    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    """Combined configuration containing model and training configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Tokenizer path
    tokenizer_path: str = "artifacts/tokenizer.model"


def load_config_from_yaml(yaml_path: str) -> Config:
    """Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract nested configs
    model_dict = config_dict.get('model', {})
    training_dict = config_dict.get('training', {})
    
    # Create config objects
    model_config = ModelConfig(**model_dict)
    training_config = TrainingConfig(**training_dict)
    
    # Handle top-level fields
    tokenizer_path = config_dict.get('tokenizer_path', 'artifacts/tokenizer.model')
    
    return Config(
        model=model_config,
        training=training_config,
        tokenizer_path=tokenizer_path
    )


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Config object to save
        output_path: Path where to save the configuration
    """
    config_dict = {
        'model': asdict(config.model),
        'training': asdict(config.training),
        'tokenizer_path': config.tokenizer_path
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def merge_config_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    """Merge configuration overrides into an existing config.
    
    Args:
        config: Base configuration
        overrides: Dictionary of overrides in dot notation
                   e.g., {"model.n_layers": 4, "training.batch_size": 2}
    
    Returns:
        Updated Config object
    """
    config_dict = {
        'model': asdict(config.model),
        'training': asdict(config.training),
        'tokenizer_path': config.tokenizer_path
    }
    
    for key, value in overrides.items():
        parts = key.split('.')
        
        if len(parts) == 1:
            # Top-level config
            config_dict[key] = value
        elif len(parts) == 2:
            # Nested config (e.g., model.n_layers)
            section, field = parts
            if section in config_dict and isinstance(config_dict[section], dict):
                config_dict[section][field] = value
            else:
                raise ValueError(f"Invalid config override key: {key}")
        else:
            raise ValueError(f"Config override key too deeply nested: {key}")
    
    # Reconstruct config
    model_config = ModelConfig(**config_dict['model'])
    training_config = TrainingConfig(**config_dict['training'])
    
    return Config(
        model=model_config,
        training=training_config,
        tokenizer_path=config_dict['tokenizer_path']
    )


def parse_override_string(override_str: str) -> tuple[str, Any]:
    """Parse a single override string in the format 'key=value'.
    
    Args:
        override_str: String like "model.n_layers=4"
        
    Returns:
        Tuple of (key, value) with value converted to appropriate type
    """
    if '=' not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected 'key=value'")
    
    key, value_str = override_str.split('=', 1)
    key = key.strip()
    value_str = value_str.strip()
    
    # Try to parse as JSON to handle different types
    try:
        value = json.loads(value_str)
    except json.JSONDecodeError:
        # If not valid JSON, treat as string
        value = value_str
    
    return key, value
