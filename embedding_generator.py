"""
Файл обработки входящего текста и создания на его основе эмбеддинга.

В этом модуле:
  - Загружается предобученная модель CLIP для получения эмбеддингов.
  - Используются NLP модели spaCy для обработки текста.
  - Реализованы функции для объединения текста с дополнительной информацией.
  - Эмбеддинг сохраняется на диск для последующего анализа.
"""

import clip
import torch
import numpy as np
import os
import datetime
import torch.nn as nn
import spacy
from langdetect import detect
# from spellchecker import SpellChecker  # Не используется в текущей версии

def EnsureScalar(value):
    """
    Преобразует numpy-массив или число в скаляр.
    """
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Value must be a scalar.")
        return value.item()
    elif isinstance(value, (int, float)):
        return value
    else:
        raise TypeError("Value must be a scalar number.")

class EmbeddingGenerator:
    """
    Генератор эмбеддингов на основе предобученной модели CLIP.
    
    При инициализации:
      - Загружается модель CLIP (ViT-B/32).
      - Если требуется уменьшение размерности эмбеддингов (reduced_dim != 512),
        добавляется линейный слой.
      - Загружаются модели spaCy для обработки текстов на русском и английском языках.
    """
    def __init__(self, device: torch.device, reduced_dim: int = 512):
        self.device = device
        self.reduced_dim = reduced_dim
        
        # Загрузка модели CLIP
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("[EMBED] Инициализация генератора эмбеддингов с использованием CLIP завершена.")
        if self.model is not None:
            print(f"[EMBED] CLIP модель загружена: {self.model}")
        
        # Если требуется уменьшение размерности, добавляем линейный слой
        if self.reduced_dim != 512:
            self.reduce_dim_layer = nn.Linear(512, self.reduced_dim).to(self.device)
            print(f"[EMBED] Добавлен линейный слой для уменьшения размерности до {self.reduced_dim}.")

        # Загрузка моделей spaCy для обработки текста (NLP)
        try:
            self.models = {
                "ru": spacy.load("ru_core_news_sm"),
                "en": spacy.load("en_core_web_sm"),
            }
            print("[EMBED] Модели обработки текста spaCy загружены успешно.")
        except Exception as e:
            print(f"[EMBED] Ошибка загрузки моделей spaCy: {e}")
            print("Установите модели командой: python -m spacy download ru_core_news_sm en_core_web_sm")
            raise

        # self.spell = SpellChecker()  # На данный момент не используется

    def extract_keywords(self, text: str):
        """
        Извлекает ключевые слова и именованные сущности из текста.
        Использует langdetect для определения языка и соответствующую модель spaCy.
        """
        lang = detect(text)
        print(f"[EMBED] Определён язык: {lang}")
        nlp = self.models.get(lang)
        if not nlp:
            raise ValueError(f"[ERROR] Для языка '{lang}' модель не загружена...")
        doc = nlp(text)
        keywords = []
        for ent in doc.ents:
            keywords.append((ent.text, ent.label_))
        for token in doc:
            if token.is_alpha and (not token.is_stop):
                keywords.append((token.text, "KEYWORD"))
        return keywords

    def highlight_keywords(self, text: str):
        """
        Возвращает текст с выделенными ключевыми словами и список ключевых слов.
        """
        keywords = self.extract_keywords(text)
        highlighted_text = text
        for keyword, _ in keywords:
            highlighted_text = highlighted_text.replace(keyword, f"[{keyword}]")
        return highlighted_text, keywords

    def combine_text(self, text: str, additional_info: str, shape_info: dict) -> str:
        """
        Объединяет основной текст с дополнительной информацией и данными о форме.
        """
        combined_text = f"{text}. {additional_info}"
        if shape_info:
            shape_description = (
                f"Размер: {shape_info.get('size', 'не указан')}, "
                f"Форма: {shape_info.get('shape', 'не указана')}, "
                f"Ориентация: {shape_info.get('orientation', 'не указана')}"
            )
            combined_text = f"{combined_text}. {shape_description}"
        return combined_text

    def generate_embedding(self, text: str, additional_info: str = "", shape_info: dict = None) -> torch.Tensor:
        """
        Генерирует эмбеддинг для заданного текста, объединяя его с дополнительной информацией.
        
        Аргументы:
          text: Основной текстовый prompt.
          additional_info: Дополнительная информация для объединения.
          shape_info: Словарь с информацией о размере, форме и ориентации.
        
        Возвращает:
          Тензор с эмбеддингом.
        """
        combined_text = self.combine_text(text, additional_info, shape_info)
        print(f"[EMBED] Генерация эмбеддинга для текста: '{combined_text}'")
        text_input = clip.tokenize([combined_text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        print(f"[EMBED] Эмбеддинг CLIP сгенерирован. Размерность: {text_features.shape}")
        if hasattr(self, 'reduce_dim_layer'):
            text_features = self.reduce_dim_layer(text_features)
            print(f"[EMBED] Размерность эмбеддинга уменьшена до {text_features.shape[1]}.")
        embedding_filepath = self.save_embedding(text_features)
        if embedding_filepath:
            print(f"[EMBED] Эмбеддинг сохранён: {embedding_filepath}")
        else:
            print("[EMBED] Ошибка при сохранении эмбеддинга.")
        return text_features

    def save_embedding(self, embedding: torch.Tensor, output_dir: str = "temp_emb", unique_filename: bool = True) -> str:
        """
        Сохраняет эмбеддинг в файл в указанной директории.
        
        Если файлов в директории больше 50, удаляет самый старый.
        Генерирует уникальное имя файла, если unique_filename=True.
        
        Возвращает путь к сохраненному файлу.
        """
        if not os.path.exists(output_dir):
            print(f"[EMBED] Папка {output_dir} не существует, создаём её.")
            os.makedirs(output_dir)
        
        print(f"[EMBED] Текущая рабочая директория: {os.getcwd()}")
        files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
        if len(files) >= 50:
            oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
            os.remove(os.path.join(output_dir, oldest_file))
            print(f"[EMBED] Удалён старый файл: {oldest_file}")
        
        filename = "embedding.npy"
        if unique_filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embedding_{timestamp}.npy"
        filepath = os.path.join(output_dir, filename)
        print(f"[EMBED] Сохранение эмбеддинга в файл: {filepath}")
        try:
            np.save(filepath, embedding.cpu().detach().numpy())
            print(f"[EMBED] Эмбеддинг успешно сохранён: {filepath}")
        except Exception as e:
            print(f"[EMBED] Ошибка при сохранении файла: {e}")
            return None
        return filepath
