import clip
import torch
import numpy as np
import os
import datetime

class EmbeddingGenerator:
    def __init__(self, device: torch.device):
        self.device = device
        # Загружаем модель CLIP
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("Инициализация генератора эмбеддингов с использованием CLIP завершена.")

    def generate_embedding(self, text: str, additional_info: str = "", shape_info: dict = None) -> torch.Tensor:
        """
        Генерирует эмбеддинг для одного текста с использованием CLIP.
        Дополнительная информация будет добавлена к тексту перед генерацией эмбеддинга.
        """
        # Добавляем дополнительную информацию и параметры формы
        combined_text = self.combine_text(text, additional_info, shape_info)
        
        # Подготовка текста
        text_input = clip.tokenize([combined_text]).to(self.device)
        
        # Генерация эмбеддинга
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        
        print(f"Генерация эмбеддинга с CLIP завершена. Размерность: {text_features.shape}")
        
        # Сохраняем эмбеддинг
        embedding_filepath = self.save_embedding(text_features)
        if embedding_filepath:
            print(f"Эмбеддинг успешно сохранён в файл: {embedding_filepath}")
        else:
            print("Ошибка при сохранении эмбеддинга.")
        return text_features

    def combine_text(self, text: str, additional_info: str, shape_info: dict) -> str:
        """
        Объединяет основной текст с дополнительной информацией и параметрами формы.
        """
        # Добавляем дополнительную информацию к тексту
        combined_text = f"{text}. {additional_info}"
        
        # Добавляем информацию о форме, если она предоставлена
        if shape_info:
            shape_description = f"Размер: {shape_info.get('size', 'не указан')}, Форма: {shape_info.get('shape', 'не указана')}, Ориентация: {shape_info.get('orientation', 'не указана')}"
            combined_text = f"{combined_text}. {shape_description}"
        
        print(f"Комбинированный текст для эмбеддинга: {combined_text}")
        return combined_text

    def save_embedding(self, embedding: torch.Tensor, output_dir: str = "temp_emb", unique_filename: bool = True) -> str:
        """
        Сохраняет эмбеддинг в файл .npy в папку temp_emb, ограничивая количество файлов до 10.
        """
        # Убедимся, что папка существует
        if not os.path.exists(output_dir):
            print(f"Папка {output_dir} не существует, создаём её.")
            os.makedirs(output_dir)

        # Проверка текущей рабочей директории
        print(f"Текущая рабочая директория: {os.getcwd()}")

        # Получаем список файлов в директории
        files = os.listdir(output_dir)
        files = [f for f in files if f.endswith('.npy')]  # фильтруем только .npy файлы

        # Если файлов больше 10, удаляем старые
        if len(files) >= 10:
            oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))  # находим старейший файл
            os.remove(os.path.join(output_dir, oldest_file))  # удаляем его
            print(f"Удалён старый файл: {oldest_file}")

        # Генерация уникального имени файла
        filename = "embedding.npy"
        if unique_filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embedding_{timestamp}.npy"

        filepath = os.path.join(output_dir, filename)

        # Проверка пути сохранения
        print(f"Путь для сохранения эмбеддинга: {filepath}")

        try:
            np.save(filepath, embedding.cpu().numpy())  # Переводим на CPU перед сохранением
            print(f'Эмбеддинг успешно сохранён: {filepath}')
        except Exception as e:
            print(f'Ошибка при сохранении файла: {e}')
            return None

        return filepath
