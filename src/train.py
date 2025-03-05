import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import wandb
import os
from tqdm import tqdm
import logging
from datetime import datetime

from text_encoder import TextEncoder
from diffusion import UNet3DConditioned, DiffusionModel
from decoder_3d import Decoder3D, NeRFDecoder
from renderer import VolumeRenderer
from configs.default import Config

class Trainer:
    """Класс для обучения модели."""
    
    def __init__(
        self,
        config: Config,
        device: torch.device,
        resume_from: Optional[str] = None
    ):
        """
        Args:
            config: Конфигурация
            device: Устройство для обучения
            resume_from: Путь к чекпоинту для продолжения обучения
        """
        self.config = config
        self.device = device
        
        # Инициализация логгера
        self.setup_logging()
        
        # Инициализация моделей
        self.setup_models()
        
        # Инициализация оптимизатора и планировщика
        self.setup_optimizer()
        
        # Загрузка чекпоинта если нужно
        if resume_from:
            self.load_checkpoint(resume_from)
            
        # Инициализация W&B
        if self.config.wandb_project:
            self.setup_wandb()
            
    def setup_logging(self):
        """Настройка логирования."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """Инициализация моделей."""
        # Текстовый энкодер
        self.text_encoder = TextEncoder(
            device=self.device,
            model_name=self.config.text_encoder.model_name,
            max_length=self.config.text_encoder.max_length
        )
        
        # U-Net
        self.unet = UNet3DConditioned(
            **vars(self.config.unet)
        ).to(self.device)
        
        # Диффузионная модель
        self.diffusion = DiffusionModel(
            model=self.unet,
            **vars(self.config.diffusion),
            device=self.device
        )
        
        # 3D декодер
        self.decoder_3d = Decoder3D(
            **vars(self.config.decoder_3d)
        ).to(self.device)
        
        # NeRF декодер
        self.nerf = NeRFDecoder(
            **vars(self.config.nerf)
        ).to(self.device)
        
        # Рендерер
        self.renderer = VolumeRenderer(
            **vars(self.config.renderer)
        ).to(self.device)
        
    def setup_optimizer(self):
        """Инициализация оптимизатора и планировщика."""
        # Собираем все параметры
        parameters = list(self.unet.parameters())
        parameters.extend(self.decoder_3d.parameters())
        parameters.extend(self.nerf.parameters())
        
        # Создаем оптимизатор
        self.optimizer = optim.AdamW(
            parameters,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Создаем планировщик с разогревом
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=self.config.training.num_epochs,
            pct_start=0.1
        )
        
    def setup_wandb(self):
        """Инициализация W&B."""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=vars(self.config)
        )
        
    def save_checkpoint(self, epoch: int, step: int):
        """
        Сохранение чекпоинта.
        
        Args:
            epoch: Номер эпохи
            step: Номер шага
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'unet_state': self.unet.state_dict(),
            'decoder_3d_state': self.decoder_3d.state_dict(),
            'nerf_state': self.nerf.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}_step_{step}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Сохранен чекпоинт: {checkpoint_path}')
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Загрузка чекпоинта.
        
        Args:
            checkpoint_path: Путь к чекпоинту
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint['unet_state'])
        self.decoder_3d.load_state_dict(checkpoint['decoder_3d_state'])
        self.nerf.load_state_dict(checkpoint['nerf_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['step']
        
        self.logger.info(f'Загружен чекпоинт: {checkpoint_path}')
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Один шаг обучения.
        
        Args:
            batch: Батч данных
            
        Returns:
            Словарь с метриками
        """
        # Получаем данные
        text = batch['text']
        mesh = batch['mesh']
        
        # Кодируем текст
        text_embeddings = self.text_encoder.encode(text)
        
        # Шаг диффузии
        diffusion_loss = self.diffusion.train_step(mesh, text_embeddings)
        
        # Декодируем в 3D
        occupancy, features = self.decoder_3d(mesh)
        reconstruction_loss = F.mse_loss(occupancy, mesh)
        
        # Рендерим виды
        colors, depths = self.renderer.render_views(self.nerf, features)
        rendering_loss = F.mse_loss(colors, batch['views'])
        
        # Суммарная функция потерь
        total_loss = (
            diffusion_loss +
            reconstruction_loss +
            rendering_loss
        )
        
        # Оптимизация
        self.optimizer.zero_grad()
        total_loss.backward()
        
        if self.config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(),
                self.config.training.gradient_clip
            )
            
        self.optimizer.step()
        
        return {
            'diffusion_loss': diffusion_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'rendering_loss': rendering_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Обучение модели.
        
        Args:
            train_loader: Загрузчик обучающих данных
            val_loader: Загрузчик валидационных данных
        """
        self.logger.info('Начало обучения...')
        
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f'Эпоха {epoch + 1}/{self.config.training.num_epochs}')
            
            # Обучение
            self.unet.train()
            self.decoder_3d.train()
            self.nerf.train()
            
            train_metrics = []
            
            for batch in tqdm(train_loader, desc='Обучение'):
                # Переносим данные на устройство
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Шаг обучения
                metrics = self.train_step(batch)
                train_metrics.append(metrics)
                
                # Логирование
                if self.config.wandb_project:
                    wandb.log(metrics)
                    
            # Сохранение чекпоинта
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(epoch + 1, self.global_step)
                
            # Валидация
            if val_loader and (epoch + 1) % self.config.training.eval_interval == 0:
                self.evaluate(val_loader)
                
            # Обновляем планировщик
            self.scheduler.step()
            
        self.logger.info('Обучение завершено!')
        
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Оценка модели на валидационном наборе.
        
        Args:
            val_loader: Загрузчик валидационных данных
            
        Returns:
            Словарь с метриками
        """
        self.logger.info('Начало валидации...')
        
        self.unet.eval()
        self.decoder_3d.eval()
        self.nerf.eval()
        
        val_metrics = []
        
        for batch in tqdm(val_loader, desc='Валидация'):
            # Переносим данные на устройство
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Получаем метрики
            metrics = self.train_step(batch)
            val_metrics.append(metrics)
            
        # Усредняем метрики
        avg_metrics = {
            k: sum(m[k] for m in val_metrics) / len(val_metrics)
            for k in val_metrics[0].keys()
        }
        
        # Логирование
        if self.config.wandb_project:
            wandb.log({'val_' + k: v for k, v in avg_metrics.items()})
            
        self.logger.info(f'Метрики валидации: {avg_metrics}')
        
        return avg_metrics
        
def main():
    """Основная функция для запуска обучения."""
    # Загружаем конфигурацию
    config = Config()
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем тренер
    trainer = Trainer(config, device)
    
    # TODO: Создать и загрузить датасет
    train_loader = None
    val_loader = None
    
    # Запускаем обучение
    trainer.train(train_loader, val_loader)
    
if __name__ == '__main__':
    main() 