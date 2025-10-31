from dataclasses import dataclass, field
from typing import List

@dataclass
class BaseModelConfig:
    """Configuration for base model architecture."""
    num_labels: int = 2
    dropout: float = 0.1

@dataclass
class TextModelConfig(BaseModelConfig):
    """Configuration for text model architecture.""" 
    text_encoder: str = "bert-base-uncased"

@dataclass
class AudioModelConfig(BaseModelConfig):
    """Configuration for audio model architecture."""
    audio_encoder: str = "openai/whisper-small"

@dataclass
class MultimodalModelConfig(TextModelConfig, AudioModelConfig):
    """Configuration for multimodal model architecture."""
    cross_attention_heads: int = 12
    use_ffn: bool = False
    
@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_path: str = "./data/ADReSS/Train"
    test_path: str = "./data/ADReSS/Test"
    val_split: float = 0.35
    audio_chunk_duration: int = 30  # seconds
    max_text_length: int = 512
    preprocess_batch_size: int = 200
    num_preprocess_workers: int = 1
    
@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./results"
    run_name: str = "cross_attention_baseline"
    
    # Training hyperparameters
    num_train_epochs: int = 50
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    
    # Optimizer and scheduler
    weight_decay: float = 0.01
    warmup_steps: int = 0
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 1
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 20
    
    # Logging
    logging_steps: int = 1
    report_to: str = "wandb"
    verbose: bool = False
    
    # Misc
    seeds: List[int] = field(default_factory=lambda: [42, 2023, 2024, 88, 1234])
    device: str = "cuda"
    overwrite_output_dir: bool = True
    
@dataclass
class Config:
    """Main configuration combining all configs."""
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)