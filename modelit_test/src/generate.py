#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
import trimesh
from typing import Tuple, Optional, List, Dict, Any, Union
from PIL import Image
import numpy as np
import time
import re

from src.text_encoder import TextEncoder
from src.diffusion import UNet3DConditioned, DiffusionModel
from src.decoder_3d import Decoder3D, NeRFDecoder
from src.renderer import VolumeRenderer


class Generator:
    """
    Генератор 3D-моделей из текстовых описаний.
    """
    
    def __init__(
        self,
        config: Any,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        Инициализация генератора.
        
        Args:
            config: Конфигурация моделей и параметров генерации.
            device: Устройство для вычислений (CPU/GPU).
            checkpoint_path: Путь к чекпойнту с весами моделей.
            log_level: Уровень логирования.
        """
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Настройка логирования
        self.setup_logging(log_level)
        
        # Инициализация моделей
        self.setup_models()
        
        # Загрузка весов из чекпойнта, если указан
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        
    def setup_logging(self, log_level: int) -> None:
        """
        Настройка логирования.
        
        Args:
            log_level: Уровень логирования.
        """
        self.logger = logging.getLogger("modelit.generator")
        self.logger.setLevel(log_level)
        
        # Проверяем, настроен ли уже логгер
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def setup_models(self) -> None:
        """Инициализация моделей для генерации."""
        self.logger.info("Инициализация моделей")
        
        # Текстовый энкодер
        self.text_encoder = TextEncoder(
            device=self.device,
            model_name=self.config.text_encoder.model_name,
            max_length=self.config.text_encoder.max_length
        )
        
        # U-Net модель
        self.unet = UNet3DConditioned(
            in_channels=self.config.unet.in_channels,
            out_channels=self.config.unet.out_channels,
            model_channels=self.config.unet.model_channels,
            num_res_blocks=self.config.unet.num_res_blocks,
            attention_resolutions=self.config.unet.attention_resolutions,
            context_dim=self.config.unet.context_dim
        )
        
        # Модель диффузии
        self.diffusion = DiffusionModel(
            unet=self.unet,
            device=self.device,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            n_timesteps=self.config.diffusion.n_timesteps,
            latent_shape=self.config.diffusion.latent_shape
        )
        
        # 3D-декодер
        self.decoder_3d = Decoder3D(
            in_channels=self.config.decoder_3d.in_channels,
            out_channels=self.config.decoder_3d.out_channels,
            model_channels=self.config.decoder_3d.model_channels,
            num_upsamples=self.config.decoder_3d.num_upsamples,
            dropout=self.config.decoder_3d.dropout
        )
        
        # NeRF-декодер
        self.nerf_decoder = NeRFDecoder(
            in_features=self.config.nerf.in_features,
            hidden_dim=self.config.nerf.hidden_dim,
            num_layers=self.config.nerf.num_layers,
            position_encode_dim=self.config.nerf.position_encode_dim,
            direction_encode_dim=self.config.nerf.direction_encode_dim
        )
        
        # Рендерер
        self.renderer = VolumeRenderer(
            device=self.device,
            resolution=self.config.renderer.resolution,
            threshold=self.config.renderer.threshold,
            camera_dist=self.config.renderer.camera_dist,
            num_views=self.config.renderer.num_views
        )
        
        # Перемещаем модели на устройство
        self.unet.to(self.device)
        self.decoder_3d.to(self.device)
        self.nerf_decoder.to(self.device)
        
        # Переводим модели в режим оценки
        self.unet.eval()
        self.decoder_3d.eval()
        self.nerf_decoder.eval()
        
        self.logger.info("Модели инициализированы")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Загрузка весов моделей из чекпойнта.
        
        Args:
            checkpoint_path: Путь к чекпойнту.
        """
        self.logger.info(f"Загрузка весов моделей из чекпойнта: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Загружаем веса модели U-Net
            if 'unet' in checkpoint:
                self.unet.load_state_dict(checkpoint['unet'])
            
            # Загружаем веса 3D-декодера
            if 'decoder_3d' in checkpoint:
                self.decoder_3d.load_state_dict(checkpoint['decoder_3d'])
            
            # Загружаем веса NeRF-декодера
            if 'nerf_decoder' in checkpoint:
                self.nerf_decoder.load_state_dict(checkpoint['nerf_decoder'])
            
            self.logger.info("Веса моделей успешно загружены")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке весов: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        save_mesh: bool = True,
        save_preview: bool = True,
        guidance_scale: float = 7.5,
        batch_size: int = 1
    ) -> Tuple[str, str]:
        """
        Генерация 3D-модели из текстового описания.
        
        Args:
            prompt: Текстовое описание модели.
            output_dir: Директория для сохранения результатов. По умолчанию используется директория из конфигурации.
            save_mesh: Флаг сохранения меша.
            save_preview: Флаг сохранения превью.
            guidance_scale: Весовой коэффициент для классификатор-свободного руководства.
            batch_size: Размер батча.
            
        Returns:
            Кортеж (путь к мешу, путь к превью).
        """
        start_time = time.time()
        self.logger.info(f"Начало генерации модели для запроса: '{prompt}'")
        
        # Создаем директорию для сохранения результатов, если не указана
        if output_dir is None:
            output_dir = self.config.generation.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Генерируем безопасное имя файла из запроса
        safe_prompt = re.sub(r'[^\w\s-]', '', prompt).strip().lower()
        safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_base = f"{safe_prompt}_{timestamp}"
        
        # Пути для сохранения файлов
        mesh_path = os.path.join(output_dir, f"{filename_base}.obj")
        preview_path = os.path.join(output_dir, f"{filename_base}.gif")
        
        # Кодирование текстового запроса
        self.logger.info("Кодирование текстового запроса")
        with torch.no_grad():
            text_embeddings = self.text_encoder.encode([prompt] * batch_size)
        
        # Генерация латентного представления с помощью диффузионной модели
        self.logger.info("Генерация латентного представления")
        with torch.no_grad():
            latents = self.diffusion.sample(
                batch_size=batch_size,
                context=text_embeddings,
                channels=self.config.diffusion.in_channels,
                guidance_scale=guidance_scale
            )
        
        # Декодирование латентного представления в воксельную сетку
        self.logger.info("Декодирование латентного представления в воксельную сетку")
        with torch.no_grad():
            voxels = self.decoder_3d.decode_latents(
                latents,
                threshold=self.config.decoder_3d.threshold
            )
        
        # Извлечение меша и рендеринг превью
        self.logger.info("Извлечение полигональной сетки и рендеринг превью")
        mesh, views = self.renderer.extract_mesh_from_voxels(voxels, smooth=True)
        
        # Сохранение меша
        if save_mesh:
            self.logger.info(f"Сохранение меша в {mesh_path}")
            self.renderer.save_mesh(mesh, mesh_path, file_type='obj')
        
        # Сохранение превью
        if save_preview:
            self.logger.info(f"Сохранение превью в {preview_path}")
            self.renderer.create_gif(views, preview_path, duration=100)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Генерация завершена за {elapsed_time:.2f} секунд")
        
        return mesh_path, preview_path


def main():
    """
    Основная функция для запуска генератора из командной строки.
    """
    import argparse
    from configs.default import Config
    from utils.utils import setup_device
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Генерация 3D-моделей из текстовых описаний")
    parser.add_argument("--prompt", type=str, required=True, help="Текстовое описание модели")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпойнту модели")
    parser.add_argument("--output_dir", type=str, default=None, help="Директория для сохранения результатов")
    parser.add_argument("--no_mesh", action="store_true", help="Не сохранять меш")
    parser.add_argument("--no_preview", action="store_true", help="Не сохранять превью")
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = Config()
    
    # Настройка устройства
    device = setup_device(cuda=config.use_cuda)
    
    # Создание генератора
    generator = Generator(
        config=config,
        device=device,
        checkpoint_path=args.checkpoint
    )
    
    # Генерация модели
    mesh_path, preview_path = generator.generate(
        prompt=args.prompt,
        output_dir=args.output_dir,
        save_mesh=not args.no_mesh,
        save_preview=not args.no_preview
    )
    
    print(f"Меш сохранен в: {mesh_path}")
    print(f"Превью сохранено в: {preview_path}")


if __name__ == "__main__":
    main() 