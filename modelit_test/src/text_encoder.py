#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from typing import List, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder:
    """Текстовый энкодер на основе CLIP для преобразования текстовых запросов в эмбеддинги."""
    
    def __init__(
        self,
        device: torch.device,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77
    ):
        """
        Инициализация текстового энкодера.
        
        Args:
            device: Устройство для выполнения вычислений (CPU/GPU).
            model_name: Название предобученной модели CLIP.
            max_length: Максимальная длина токенизированного запроса.
        """
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        
        # Загрузка токенизатора и модели
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        
        # Переводим модель в режим оценки
        self.model.eval()
    
    def encode(
        self,
        prompts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Кодирование текстовых запросов в эмбеддинги.
        
        Args:
            prompts: Список текстовых запросов.
            normalize: Флаг нормализации эмбеддингов по L2 норме.
            
        Returns:
            Тензор эмбеддингов формы [batch_size, embedding_dim].
        """
        # Токенизация запросов
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Перемещаем данные на устройство
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        # Получаем эмбеддинги с отключенным градиентом
        with torch.no_grad():
            text_embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]
        
        # Нормализация по L2 норме, если требуется
        if normalize:
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings
    
    def encode_pooled(
        self,
        prompts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Кодирование текстовых запросов в пулированные эмбеддинги.
        
        Args:
            prompts: Список текстовых запросов.
            normalize: Флаг нормализации эмбеддингов по L2 норме.
            
        Returns:
            Тензор пулированных эмбеддингов формы [batch_size, embedding_dim].
        """
        # Токенизация запросов
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Перемещаем данные на устройство
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        # Получаем пулированные эмбеддинги с отключенным градиентом
        with torch.no_grad():
            text_embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[1]
        
        # Нормализация по L2 норме, если требуется
        if normalize:
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings 