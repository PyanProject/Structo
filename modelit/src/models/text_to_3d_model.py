import torch
import torch.nn as nn
import os
import sys

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .text_encoder import TextEncoder
from .shape_generator import VoxelTransformer

class TextTo3DModel(nn.Module):
    """
    Полная модель для генерации 3D объектов из текстовых описаний.
    """
    
    def __init__(self, config):
        """
        Инициализация модели Text-to-3D.
        
        Args:
            config: Конфигурация модели.
        """
        super(TextTo3DModel, self).__init__()
        
        # Инициализация текстового энкодера
        self.text_encoder = TextEncoder(
            pretrained=config.model.text_encoder.pretrained,
            freeze=config.model.text_encoder.freeze,
            embedding_dim=config.model.text_encoder.embedding_dim
        )
        
        # Инициализация генератора 3D форм
        self.shape_generator = VoxelTransformer(
            latent_dim=config.model.shape_generator.latent_dim,
            hidden_dims=config.model.shape_generator.hidden_dims,
            dropout=config.model.shape_generator.dropout,
            num_heads=config.model.shape_generator.num_heads,
            num_layers=config.model.shape_generator.num_layers,
            voxel_dim=config.model.shape_generator.voxel_dim
        )
        
        # Сохраняем конфигурацию
        self.config = config
    
    def forward(self, text_prompts):
        """
        Прямой проход модели.
        
        Args:
            text_prompts (List[str]): Список текстовых описаний.
            
        Returns:
            torch.Tensor: Воксельное представление 3D объектов.
        """
        # Получение текстовых эмбеддингов
        text_embeddings = self.text_encoder(text_prompts)
        
        # Генерация 3D объектов
        voxel_grid = self.shape_generator(text_embeddings)
        
        return voxel_grid
    
    def generate(self, text_prompt, threshold=0.5):
        """
        Генерация 3D объекта по текстовому описанию.
        
        Args:
            text_prompt (str): Текстовое описание объекта.
            threshold (float): Порог для бинаризации воксельной сетки.
            
        Returns:
            torch.Tensor: Бинаризованное воксельное представление 3D объекта.
        """
        # Переключаем модель в режим оценки
        self.eval()
        
        with torch.no_grad():
            # Генерация воксельной сетки
            voxel_grid = self.forward([text_prompt])
            
            # Бинаризация по порогу
            binary_voxel_grid = (torch.sigmoid(voxel_grid) > threshold).float()
        
        return binary_voxel_grid
    
    def enable_gradient_checkpointing(self):
        """
        Включает gradient checkpointing для экономии памяти.
        Это замедляет обучение, но значительно снижает использование памяти.
        """
        # Включаем gradient checkpointing в генераторе формы
        if hasattr(self.shape_generator, 'use_gradient_checkpointing'):
            self.shape_generator.use_gradient_checkpointing = True
            print("Включен gradient checkpointing для генератора формы")
        
        # Включаем gradient checkpointing в текстовом энкодере, если он не заморожен
        if not self.config.model.text_encoder.freeze and hasattr(self.text_encoder, 'model'):
            if hasattr(self.text_encoder.model, 'gradient_checkpointing_enable'):
                self.text_encoder.model.gradient_checkpointing_enable()
                print("Включен gradient checkpointing для текстового энкодера")
        
        print("Gradient checkpointing успешно настроен для экономии памяти")
    
    def get_memory_usage(self):
        """
        Возвращает текущее использование памяти моделью.
        
        Returns:
            dict: Словарь с информацией об использовании памяти.
        """
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats['allocated'] = torch.cuda.memory_allocated() / 1e9  # В ГБ
            memory_stats['cached'] = torch.cuda.memory_reserved() / 1e9  # В ГБ
            memory_stats['max_allocated'] = torch.cuda.max_memory_allocated() / 1e9  # В ГБ
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        memory_stats['total_params'] = total_params
        memory_stats['trainable_params'] = trainable_params
        memory_stats['frozen_params'] = total_params - trainable_params
        
        return memory_stats 