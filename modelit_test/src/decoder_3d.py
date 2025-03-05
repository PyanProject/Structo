#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

class Decoder3D(torch.nn.Module):
    """
    Декодер для преобразования латентных представлений в трехмерные вокселы.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        model_channels: int = 64,
        num_upsamples: int = 3,
        dropout: float = 0.0,
    ):
        """
        Инициализация декодера.
        
        Args:
            in_channels: Количество каналов на входе.
            out_channels: Количество каналов на выходе.
            model_channels: Базовое количество каналов в модели.
            num_upsamples: Количество слоев повышения разрешения.
            dropout: Вероятность отключения нейронов (dropout).
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_upsamples = num_upsamples
        self.dropout = dropout
        
        # Создаем блоки декодера
        # Входной блок
        self.input_block = torch.nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Последовательно создаем блоки повышения разрешения
        self.upsampling_blocks = torch.nn.ModuleList()
        current_channels = model_channels
        for i in range(num_upsamples):
            # Каждый следующий блок имеет вдвое меньше каналов
            next_channels = current_channels // 2
            self.upsampling_blocks.append(
                self._create_upsampling_block(current_channels, next_channels)
            )
            current_channels = next_channels
        
        # Выходной блок
        self.output_block = torch.nn.Sequential(
            torch.nn.Conv3d(current_channels, current_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(current_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(current_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def _create_upsampling_block(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """
        Создание блока повышения разрешения.
        
        Args:
            in_channels: Количество входных каналов.
            out_channels: Количество выходных каналов.
            
        Returns:
            Блок повышения разрешения.
        """
        return torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout3d(p=self.dropout),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через декодер.
        
        Args:
            x: Входной тензор формы [batch_size, in_channels, depth, height, width].
            
        Returns:
            Выходной тензор формы [batch_size, out_channels, depth*scale, height*scale, width*scale],
            где scale = 2^num_upsamples.
        """
        # Входной блок
        h = self.input_block(x)
        
        # Проход через блоки повышения разрешения
        for upsampling_block in self.upsampling_blocks:
            h = upsampling_block(h)
        
        # Выходной блок
        output = self.output_block(h)
        
        return output
    
    def decode_latents(self, latents: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Декодирование латентных представлений в бинарные вокселы.
        
        Args:
            latents: Латентные представления формы [batch_size, in_channels, depth, height, width].
            threshold: Порог для бинаризации значений.
            
        Returns:
            Бинарное воксельное представление формы [batch_size, 1, depth*scale, height*scale, width*scale].
        """
        # Получаем непрерывные выходные значения
        continuous_output = self.forward(latents)
        
        # Применяем сигмоиду для приведения значений к диапазону [0, 1]
        continuous_output = torch.sigmoid(continuous_output)
        
        # Бинаризуем значения
        binary_output = (continuous_output > threshold).float()
        
        return binary_output


class NeRFDecoder(torch.nn.Module):
    """
    Декодер NeRF (Neural Radiance Fields) для представления трехмерных объектов.
    """
    
    def __init__(
        self,
        in_features: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        position_encode_dim: int = 10,
        direction_encode_dim: int = 4,
    ):
        """
        Инициализация декодера NeRF.
        
        Args:
            in_features: Количество входных фичей.
            hidden_dim: Размерность скрытых слоев.
            num_layers: Количество слоев MLP.
            position_encode_dim: Размерность позиционного кодирования.
            direction_encode_dim: Размерность кодирования направления.
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.position_encode_dim = position_encode_dim
        self.direction_encode_dim = direction_encode_dim
        
        # Входной размер после позиционного кодирования
        position_encoded_dim = 3 * 2 * position_encode_dim
        
        # Создаем MLP для предсказания плотности (sigma)
        self.sigma_net = torch.nn.ModuleList()
        
        # Первый слой
        self.sigma_net.append(torch.nn.Linear(position_encoded_dim + in_features, hidden_dim))
        
        # Промежуточные слои
        for i in range(num_layers - 1):
            self.sigma_net.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        # Слой для предсказания плотности
        self.sigma_layer = torch.nn.Linear(hidden_dim, 1)
        
        # Слой для предсказания фичей
        self.feature_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Создаем MLP для предсказания цвета с учетом направления
        # Входной размер после кодирования направления
        direction_encoded_dim = 3 * 2 * direction_encode_dim
        
        # RGB сеть
        self.rgb_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + direction_encoded_dim, hidden_dim // 2),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dim // 2, 3),
            torch.nn.Sigmoid()
        )
        
    def _positional_encoding(
        self,
        x: torch.Tensor,
        num_encoding_functions: int
    ) -> torch.Tensor:
        """
        Применение позиционного кодирования к входным координатам.
        
        Args:
            x: Входной тензор координат формы [..., 3].
            num_encoding_functions: Количество частот для кодирования.
            
        Returns:
            Тензор закодированных координат.
        """
        encoding = [x]
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            device=x.device
        )
        
        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))
                
        return torch.cat(encoding, dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через NeRF.
        
        Args:
            x: Координаты точек формы [batch_size, num_points, 3].
            d: Направления формы [batch_size, num_points, 3].
            z: Латентный вектор формы [batch_size, in_features].
            
        Returns:
            Кортеж (rgb, sigma), где rgb - цвета формы [batch_size, num_points, 3],
            а sigma - плотности формы [batch_size, num_points, 1].
        """
        # Получаем размеры батча и количество точек
        batch_size, num_points, _ = x.shape
        
        # Позиционное кодирование координат
        x_encoded = self._positional_encoding(x, self.position_encode_dim)
        
        # Позиционное кодирование направлений
        d_encoded = self._positional_encoding(d, self.direction_encode_dim)
        
        # Расширяем латентный вектор для всех точек
        z_expanded = z.unsqueeze(1).expand(-1, num_points, -1)
        
        # Конкатенируем закодированные координаты и латентный вектор
        h = torch.cat([x_encoded, z_expanded], dim=-1)
        
        # Проход через MLP для предсказания плотности
        for i, layer in enumerate(self.sigma_net):
            h = layer(h)
            h = torch.nn.functional.relu(h)
        
        # Предсказание плотности
        sigma = self.sigma_layer(h)
        
        # Извлечение фичей для предсказания цвета
        features = self.feature_layer(h)
        
        # Конкатенируем фичи и закодированные направления
        rgb_input = torch.cat([features, d_encoded], dim=-1)
        
        # Предсказание цвета
        rgb = self.rgb_net(rgb_input)
        
        return rgb, sigma 