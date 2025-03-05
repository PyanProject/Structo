import torch
import pytest
import numpy as np
from src.text_encoder import TextEncoder
from src.diffusion import UNet3DConditioned, DiffusionModel
from src.decoder_3d import Decoder3D, NeRFDecoder
from src.renderer import VolumeRenderer
from configs.default import Config

@pytest.fixture
def device():
    """Фикстура для устройства."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def config():
    """Фикстура для конфигурации."""
    return Config()

@pytest.fixture
def batch_size():
    """Фикстура для размера батча."""
    return 2

def test_text_encoder(device, config, batch_size):
    """Тест текстового энкодера."""
    # Инициализация
    encoder = TextEncoder(
        device=device,
        model_name=config.text_encoder.model_name,
        max_length=config.text_encoder.max_length
    )
    
    # Тестовые данные
    prompts = ['a red chair', 'a blue car']
    
    # Проверка размерностей
    embeddings = encoder.encode(prompts)
    assert embeddings.shape[0] == batch_size
    assert embeddings.device == device
    
    # Проверка нормализации
    norms = torch.norm(embeddings, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))

def test_unet(device, config, batch_size):
    """Тест U-Net."""
    # Инициализация
    unet = UNet3DConditioned(
        **vars(config.unet)
    ).to(device)
    
    # Тестовые данные
    x = torch.randn(batch_size, config.unet.in_channels, 32, 32, 32).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    cond = torch.randn(batch_size, config.unet.condition_dim).to(device)
    
    # Проверка прямого прохода
    out = unet(x, t, cond)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

def test_diffusion(device, config, batch_size):
    """Тест диффузионной модели."""
    # Инициализация
    unet = UNet3DConditioned(
        **vars(config.unet)
    ).to(device)
    
    diffusion = DiffusionModel(
        model=unet,
        **vars(config.diffusion),
        device=device
    )
    
    # Тестовые данные
    x = torch.randn(batch_size, config.unet.in_channels, 32, 32, 32).to(device)
    cond = torch.randn(batch_size, config.unet.condition_dim).to(device)
    
    # Проверка шага обучения
    loss = diffusion.train_step(x, cond)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    
    # Проверка сэмплирования
    samples = diffusion.sample(batch_size, cond)
    assert samples.shape == x.shape
    assert not torch.isnan(samples).any()

def test_decoder_3d(device, config, batch_size):
    """Тест 3D декодера."""
    # Инициализация
    decoder = Decoder3D(
        **vars(config.decoder_3d)
    ).to(device)
    
    # Тестовые данные
    z = torch.randn(batch_size, config.decoder_3d.latent_dim).to(device)
    
    # Проверка прямого прохода
    occupancy, features = decoder(z)
    assert occupancy.shape == (batch_size, config.decoder_3d.voxel_size, config.decoder_3d.voxel_size, config.decoder_3d.voxel_size)
    assert features.shape == (batch_size, config.decoder_3d.voxel_size, config.decoder_3d.voxel_size, config.decoder_3d.voxel_size, config.decoder_3d.feature_size)
    assert not torch.isnan(occupancy).any()
    assert not torch.isnan(features).any()

def test_nerf_decoder(device, config, batch_size):
    """Тест NeRF декодера."""
    # Инициализация
    nerf = NeRFDecoder(
        **vars(config.nerf)
    ).to(device)
    
    # Тестовые данные
    num_points = 1000
    points = torch.randn(batch_size, num_points, 3).to(device)
    latent = torch.randn(batch_size, config.nerf.latent_dim).to(device)
    
    # Проверка прямого прохода
    colors, density = nerf(points, latent)
    assert colors.shape == (batch_size, num_points, 3)
    assert density.shape == (batch_size, num_points, 1)
    assert not torch.isnan(colors).any()
    assert not torch.isnan(density).any()
    assert torch.all(colors >= 0) and torch.all(colors <= 1)
    assert torch.all(density >= 0)

def test_renderer(device, config, batch_size):
    """Тест рендерера."""
    # Инициализация
    renderer = VolumeRenderer(
        **vars(config.renderer)
    ).to(device)
    
    nerf = NeRFDecoder(
        **vars(config.nerf)
    ).to(device)
    
    # Тестовые данные
    latent = torch.randn(batch_size, config.nerf.latent_dim).to(device)
    
    # Проверка рендеринга
    rgb, depth = renderer(nerf, latent)
    assert rgb.shape == (batch_size, config.renderer.image_size, config.renderer.image_size, 3)
    assert depth.shape == (batch_size, config.renderer.image_size, config.renderer.image_size)
    assert not torch.isnan(rgb).any()
    assert not torch.isnan(depth).any()
    assert torch.all(rgb >= 0) and torch.all(rgb <= 1)
    assert torch.all(depth >= renderer.min_depth) and torch.all(depth <= renderer.max_depth)

def test_end_to_end(device, config):
    """Сквозной тест всего пайплайна."""
    batch_size = 1
    prompt = "a red chair"
    
    # Инициализация моделей
    text_encoder = TextEncoder(
        device=device,
        model_name=config.text_encoder.model_name,
        max_length=config.text_encoder.max_length
    )
    
    unet = UNet3DConditioned(
        **vars(config.unet)
    ).to(device)
    
    diffusion = DiffusionModel(
        model=unet,
        **vars(config.diffusion),
        device=device
    )
    
    decoder_3d = Decoder3D(
        **vars(config.decoder_3d)
    ).to(device)
    
    nerf = NeRFDecoder(
        **vars(config.nerf)
    ).to(device)
    
    renderer = VolumeRenderer(
        **vars(config.renderer)
    ).to(device)
    
    # Прогон через весь пайплайн
    # 1. Кодирование текста
    text_embeddings = text_encoder.encode([prompt])
    assert not torch.isnan(text_embeddings).any()
    
    # 2. Генерация латентного представления
    latents = diffusion.sample(batch_size, text_embeddings)
    assert not torch.isnan(latents).any()
    
    # 3. Декодирование в 3D
    occupancy, features = decoder_3d(latents)
    assert not torch.isnan(occupancy).any()
    assert not torch.isnan(features).any()
    
    # 4. Рендеринг
    rgb, depth = renderer(nerf, latents)
    assert not torch.isnan(rgb).any()
    assert not torch.isnan(depth).any()
    
    # Проверка конечных размерностей
    assert rgb.shape == (batch_size, config.renderer.image_size, config.renderer.image_size, 3)
    assert depth.shape == (batch_size, config.renderer.image_size, config.renderer.image_size) 