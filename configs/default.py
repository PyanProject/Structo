"""Конфигурация по умолчанию для обучения и генерации."""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TextEncoderConfig:
    """Конфигурация текстового энкодера."""
    model_name: str = "openai/clip-vit-large-patch14"
    max_length: int = 77

@dataclass
class UNetConfig:
    """Конфигурация U-Net."""
    in_channels: int = 4
    model_channels: int = 128
    out_channels: int = 4
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (2, 4)
    dropout: float = 0.0
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 3
    num_heads: int = 8
    num_head_channels: int = 32
    num_heads_upsample: int = -1
    resblock_updown: bool = True
    use_scale_shift_norm: bool = True
    condition_dim: int = 512

@dataclass
class DiffusionConfig:
    """Конфигурация диффузионной модели."""
    beta_start: float = 0.0001
    beta_end: float = 0.02
    n_timesteps: int = 1000

@dataclass
class Decoder3DConfig:
    """Конфигурация 3D декодера."""
    latent_dim: int = 512
    hidden_dims: List[int] = (512, 256, 128, 64)
    voxel_size: int = 32
    feature_size: int = 32

@dataclass
class NeRFConfig:
    """Конфигурация NeRF декодера."""
    latent_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 8
    num_frequencies: int = 10
    skip_connections: List[int] = (4,)

@dataclass
class RendererConfig:
    """Конфигурация рендерера."""
    image_size: int = 256
    n_pts_per_ray: int = 128
    min_depth: float = 2.0
    max_depth: float = 6.0
    stratified: bool = True

@dataclass
class TrainingConfig:
    """Конфигурация обучения."""
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    save_interval: int = 1000
    eval_interval: int = 100
    num_workers: int = 4
    mixed_precision: bool = True
    seed: int = 42

@dataclass
class GenerationConfig:
    """Конфигурация генерации."""
    batch_size: int = 1
    cfg_scale: float = 7.5
    num_inference_steps: int = 50
    output_type: str = "mesh"  # "mesh" или "pointcloud"
    num_views: int = 8
    elevation: float = 0.0
    camera_radius: float = 4.0

@dataclass
class Config:
    """Основная конфигурация."""
    text_encoder: TextEncoderConfig = TextEncoderConfig()
    unet: UNetConfig = UNetConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    decoder_3d: Decoder3DConfig = Decoder3DConfig()
    nerf: NeRFConfig = NeRFConfig()
    renderer: RendererConfig = RendererConfig()
    training: TrainingConfig = TrainingConfig()
    generation: GenerationConfig = GenerationConfig()
    
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