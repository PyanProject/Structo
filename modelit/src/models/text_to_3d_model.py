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
            binary_voxel_grid = (voxel_grid > threshold).float()
        
        return binary_voxel_grid 