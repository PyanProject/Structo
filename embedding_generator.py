import clip
import torch
import numpy as np
import os
import datetime
import torch.nn as nn

class EmbeddingGenerator:
    def __init__(self, device: torch.device, reduced_dim: int = 512):
        self.device = device
        self.reduced_dim = reduced_dim
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("[EMBED] Инициализация генератора эмбеддингов с использованием CLIP завершена.")
        if self.model is not None:
            print(f"[EMBED] CLIP модель загружена: {self.model}")
        if self.reduced_dim != 512:
            self.reduce_dim_layer = nn.Linear(512, self.reduced_dim).to(self.device)
            print(f"[EMBED] Добавлен линейный слой для уменьшения размерности до {self.reduced_dim}.")

    def generate_embedding(self, text: str, additional_info: str = "", shape_info: dict = None) -> torch.Tensor:
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

    def combine_text(self, text: str, additional_info: str, shape_info: dict) -> str:
        combined_text = f"{text}. {additional_info}"
        if shape_info:
            shape_description = f"Размер: {shape_info.get('size', 'не указан')}, Форма: {shape_info.get('shape', 'не указана')}, Ориентация: {shape_info.get('orientation', 'не указана')}"
            combined_text = f"{combined_text}. {shape_description}"
        return combined_text

    def save_embedding(self, embedding: torch.Tensor, output_dir: str = "temp_emb", unique_filename: bool = True) -> str:
        if not os.path.exists(output_dir):
            print(f"[EMBED] Папка {output_dir} не существует, создаём её.")
            os.makedirs(output_dir)
        
        print(f"[EMBED] Текущая рабочая директория: {os.getcwd()}")

        files = os.listdir(output_dir)
        files = [f for f in files if f.endswith('.npy')]
        if len(files) >= 10:
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
            np.save(filepath, embedding.cpu().numpy())
            print(f"[EMBED] Эмбеддинг успешно сохранён: {filepath}")
        except Exception as e:
            print(f"[EMBED] Ошибка при сохранении файла: {e}")
            return None
    
        return filepath
