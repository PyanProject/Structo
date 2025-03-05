#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Синусоидальные позиционные эмбеддинги для временных шагов
    """
    def __init__(self, dim: int):
        """
        Инициализирует синусоидальные позиционные эмбеддинги
        
        Args:
            dim: размерность эмбеддингов
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет позиционные эмбеддинги для заданных временных шагов
        
        Args:
            time: тензор с временными шагами [batch_size]
            
        Returns:
            Тензор с позиционными эмбеддингами [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        # Если размерность нечетная, добавляем нули для выравнивания
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
        
        return embeddings

class SelfAttention(nn.Module):
    """
    Модуль самовнимания (Self-Attention)
    """
    def __init__(self, n_heads: int, d_embed: int, dropout: float = 0.0):
        """
        Инициализирует модуль самовнимания
        
        Args:
            n_heads: количество голов внимания
            d_embed: размерность эмбеддингов
            dropout: коэффициент dropout
        """
        super().__init__()
        
        assert d_embed % n_heads == 0, "Размерность должна быть кратна количеству голов"
        
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.head_dim = d_embed // n_heads
        
        # Линейные проекции для запросов, ключей и значений
        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.v_proj = nn.Linear(d_embed, d_embed)
        
        # Выходная проекция
        self.out_proj = nn.Linear(d_embed, d_embed)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет самовнимание
        
        Args:
            x: входной тензор [batch_size, seq_len, d_embed]
            
        Returns:
            Выходной тензор после применения самовнимания [batch_size, seq_len, d_embed]
        """
        batch_size, seq_len, _ = x.shape
        
        # Вычисляем запросы, ключи, значения
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Вычисляем скоры внимания
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Применяем softmax для получения весов внимания
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Вычисляем выход внимания
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_embed)
        
        # Применяем выходную проекцию
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output

class ResidualBlock(nn.Module):
    """
    Остаточный блок трансформера (Residual Block)
    """
    def __init__(self, d_embed: int, n_heads: int, dropout: float = 0.0):
        """
        Инициализирует остаточный блок
        
        Args:
            d_embed: размерность эмбеддингов
            n_heads: количество голов внимания
            dropout: коэффициент dropout
        """
        super().__init__()
        
        # Первый слой нормализации
        self.norm1 = nn.LayerNorm(d_embed)
        
        # Слой внимания
        self.attn = SelfAttention(n_heads, d_embed, dropout)
        
        # Второй слой нормализации
        self.norm2 = nn.LayerNorm(d_embed)
        
        # Полносвязный слой
        self.ff = nn.Sequential(
            nn.Linear(d_embed, d_embed * 4),
            nn.GELU(),
            nn.Linear(d_embed * 4, d_embed),
            nn.Dropout(dropout)
        )
        
        # Dropout для выхода слоя внимания
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет выход остаточного блока
        
        Args:
            x: входной тензор [batch_size, seq_len, d_embed]
            
        Returns:
            Выходной тензор после применения остаточного блока [batch_size, seq_len, d_embed]
        """
        # Применяем слой внимания с остаточным соединением
        x = x + self.attn(self.norm1(x))
        
        # Применяем полносвязный слой с остаточным соединением
        x = x + self.ff(self.norm2(x))
        
        return x

class PointDiffusionTransformer(nn.Module):
    """
    Трансформер для диффузии облаков точек
    """
    def __init__(
        self, 
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 4096,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Инициализирует трансформер для диффузии облаков точек
        
        Args:
            device: устройство для вычислений
            dtype: тип данных
            input_channels: количество входных каналов (обычно 3 для xyz)
            output_channels: количество выходных каналов (обычно равно input_channels)
            n_ctx: максимальное количество точек
            width: размерность скрытых слоев
            layers: количество слоев трансформера
            heads: количество голов внимания
            dropout: коэффициент dropout
        """
        super().__init__()
        
        self.model_type = "point_diffusion"
        self.device = device
        self.dtype = dtype
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.width = width
        
        # Эмбеддинги для временных шагов
        self.time_embeddings = SinusoidalPositionEmbeddings(width).to(device)
        
        # Эмбеддинги для точек
        self.point_embeddings = nn.Linear(input_channels, width).to(device)
        
        # Слои трансформера
        self.transformer_blocks = nn.ModuleList([
            ResidualBlock(width, heads, dropout).to(device)
            for _ in range(layers)
        ]).to(device)
        
        # Нормализация
        self.norm = nn.LayerNorm(width).to(device)
        
        # Выходная проекция
        self.output_proj = nn.Linear(width, output_channels).to(device)
        
        # Явно переносим модель на устройство
        self.to(device)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через трансформер
        
        Args:
            x: входной тензор облака точек [batch_size, n_points, input_channels]
            t: временные шаги [batch_size]
            
        Returns:
            Выходной тензор предсказанного шума [batch_size, n_points, output_channels]
        """
        batch_size, n_points, _ = x.shape
        assert n_points <= self.n_ctx, f"Количество точек ({n_points}) превышает максимальное ({self.n_ctx})"
        
        # Эмбеддинги для точек
        h = self.point_embeddings(x)  # [batch_size, n_points, width]
        
        # Эмбеддинги для временных шагов
        time_emb = self.time_embeddings(t)  # [batch_size, width]
        time_emb = time_emb.unsqueeze(1).expand(-1, n_points, -1)  # [batch_size, n_points, width]
        
        # Добавляем временные эмбеддинги
        h = h + time_emb
        
        # Проходим через слои трансформера
        for block in self.transformer_blocks:
            h = block(h)
        
        # Нормализация и выходная проекция
        h = self.norm(h)
        output = self.output_proj(h)
        
        return output

class TextEncoder(nn.Module):
    """
    Простой текстовый энкодер
    В реальной реализации здесь был бы CLIP или другая предобученная текстовая модель
    """
    def __init__(self, output_dim: int = 512, device: Optional[torch.device] = None):
        """
        Инициализирует текстовый энкодер
        
        Args:
            output_dim: размерность выходных эмбеддингов
            device: устройство для вычислений
        """
        super().__init__()
        # Упрощенная заглушка для текстового энкодера
        self.output_dim = output_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Добавляем фиктивный параметр, чтобы модель имела параметры и могла быть перемещена на устройство
        self.dummy = nn.Parameter(torch.zeros(1, device=self.device))
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Кодирует тексты в эмбеддинги
        
        Args:
            texts: список текстовых описаний
            
        Returns:
            Тензор с текстовыми эмбеддингами [batch_size, output_dim]
        """
        # В реальной реализации здесь было бы кодирование текста с помощью предобученной модели
        # Здесь мы просто возвращаем случайные эмбеддинги
        batch_size = len(texts)
        # Явно указываем устройство
        return torch.randn(batch_size, self.output_dim, device=self.device)

class TextConditionedTransformer(PointDiffusionTransformer):
    """
    Трансформер для диффузии облаков точек с текстовыми условиями
    """
    def __init__(
        self, 
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 4096,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        dropout: float = 0.1,
        text_embed_dim: int = 512
    ):
        """
        Инициализирует трансформер с текстовыми условиями
        
        Args:
            device: устройство для вычислений
            dtype: тип данных
            input_channels: количество входных каналов (обычно 3 для xyz)
            output_channels: количество выходных каналов (обычно равно input_channels)
            n_ctx: максимальное количество точек
            width: размерность скрытых слоев
            layers: количество слоев трансформера
            heads: количество голов внимания
            dropout: коэффициент dropout
            text_embed_dim: размерность текстовых эмбеддингов
        """
        super().__init__(
            device=device,
            dtype=dtype,
            input_channels=input_channels,
            output_channels=output_channels,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            dropout=dropout
        )
        
        self.model_type = "text300M"
        
        # Текстовый энкодер - явно передаем устройство
        self.text_encoder = TextEncoder(text_embed_dim, device=device).to(device)
        
        # Проектор для текстовых эмбеддингов - явно перемещаем на устройство
        self.text_proj = nn.Linear(text_embed_dim, width).to(device)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, texts: Optional[List[str]] = None, **kwargs) -> torch.Tensor:
        """
        Прямой проход через трансформер с текстовыми условиями
        
        Args:
            x: входной тензор облака точек [batch_size, n_points, input_channels]
            t: временные шаги [batch_size]
            texts: список текстовых описаний
            
        Returns:
            Выходной тензор предсказанного шума [batch_size, n_points, output_channels]
        """
        # Убедимся, что все тензоры находятся на одном устройстве
        device = x.device
        # Убедимся, что т находится на том же устройстве
        t = t.to(device) 
        
        batch_size, n_points, _ = x.shape
        assert n_points <= self.n_ctx, f"Количество точек ({n_points}) превышает максимальное ({self.n_ctx})"
        
        # Эмбеддинги для точек
        h = self.point_embeddings(x.to(device))  # [batch_size, n_points, width]
        
        # Эмбеддинги для временных шагов
        time_emb = self.time_embeddings(t.to(device))  # [batch_size, width]
        time_emb = time_emb.unsqueeze(1).expand(-1, n_points, -1)  # [batch_size, n_points, width]
        
        # Добавляем временные эмбеддинги
        h = h + time_emb
        
        # Добавляем текстовые условия, если они предоставлены
        if texts is not None:
            # Кодируем тексты
            text_emb = self.text_encoder(texts)  # [batch_size, text_embed_dim]
            # Убедимся, что текстовые эмбеддинги находятся на том же устройстве
            text_emb = text_emb.to(device)
            # Проецируем текстовые эмбеддинги в пространство модели
            text_emb = self.text_proj(text_emb)  # [batch_size, width]
            # Добавляем текстовые эмбеддинги
            text_emb = text_emb.unsqueeze(1).expand(-1, n_points, -1)  # [batch_size, n_points, width]
            h = h + text_emb
        
        # Проходим через слои трансформера
        for block in self.transformer_blocks:
            h = block(h)
        
        # Нормализация и выходная проекция
        h = self.norm(h)
        output = self.output_proj(h)
        
        return output 