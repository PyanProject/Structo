#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Модуль с конфигурациями для моделей и обучения.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class TextEncoderConfig:
    """Конфигурация текстового энкодера."""
    model_name: str = "openai/clip-vit-base-patch32"
    max_length: int = 77
    embedding_dim: int = 512
    device: str = "cuda"

@dataclass
class UNetConfig:
    """Конфигурация UNet модели."""
    input_channels: int = 4
    base_channels: int = 256
    channel_multipliers: list = field(default_factory=lambda: [1, 2, 4, 8])
    attention_resolutions: list = field(default_factory=lambda: [8, 4, 2])
    num_heads: int = 8
    num_head_channels: int = 32
    transformer_depth: int = 1
    context_dim: int = 512
    dropout: float = 0.1
    use_checkpoint: bool = False
    legacy: bool = False
    time_embedding_dim: int = 256

@dataclass
class DiffusionConfig:
    """Конфигурация диффузионной модели."""
    beta_start: float = 1e-4
    beta_end: float = 0.02
    n_timesteps: int = 1000
    cfg_scale: float = 7.5
    prediction_type: str = "epsilon"

@dataclass
class Decoder3DConfig:
    """Конфигурация 3D декодера."""
    latent_dim: int = 512
    voxel_size: int = 64
    feature_size: int = 32
    hidden_size: int = 256
    num_layers: int = 4
    use_attention: bool = True
    use_grid_residual: bool = True
    use_residual_blocks: bool = True

@dataclass
class RendererConfig:
    """Конфигурация рендерера."""
    image_size: int = 256
    n_pts_per_ray: int = 64
    min_depth: float = 0.1
    max_depth: float = 5.0
    n_views: int = 4
    bg_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

@dataclass
class GenerationConfig:
    """Конфигурация генерации."""
    cfg_scale: float = 7.5
    num_inference_steps: int = 50
    output_type: str = "mesh"  # Может быть "mesh" или "voxels"
    num_views: int = 4
    use_deterministic_inference: bool = True
    seed: int = 42

@dataclass
class TrainingConfig:
    """Конфигурация обучения."""
    # Общие параметры обучения
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    
    # Параметры для управления процессом обучения
    log_interval: int = 10
    val_interval: int = 100
    save_interval: int = 1000
    
    # Веса различных компонентов функции потерь
    diffusion_loss_weight: float = 1.0
    decoder_loss_weight: float = 0.5
    
    # Параметры для работы с Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "modelit"
    wandb_entity: str = None
    
    # Параметры для обработки данных
    voxel_res: int = 64
    use_augmentation: bool = True
    cache_data: bool = True
    
    # Параметры для интеграции с Objaverse
    is_objaverse: bool = True
    cache_path: str = "data/objaverse/cache"
    load_in_memory: bool = False

@dataclass
class ModelConfig:
    """Общая конфигурация модели."""
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    decoder_3d: Decoder3DConfig = field(default_factory=Decoder3DConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Параметры для сериализации модели
    model_version: str = "1.0.0"
    model_name: str = "ModelIT"
    model_description: str = "Модель для генерации 3D объектов из текстовых описаний"
    
    # Параметры для инференса
    device: str = "cuda"
    use_fp16: bool = True
    use_xformers: bool = False

@dataclass
class Config:
    """Основная конфигурация."""
    text_encoder: TextEncoderConfig = TextEncoderConfig()
    unet: UNetConfig = UNetConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    decoder_3d: Decoder3DConfig = Decoder3DConfig()
    renderer: RendererConfig = RendererConfig()
    generation: GenerationConfig = GenerationConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Пути к данным и чекпоинтам
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    
    # Логирование
    wandb_project: str = "modelit"
    wandb_entity: str = None
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Проверка и обработка после инициализации."""
        import os
        
        # Создаем необходимые директории
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True) 