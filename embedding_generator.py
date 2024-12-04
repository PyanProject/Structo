import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import datetime
import os

class EmbeddingGenerator:
    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)  # Переносим модель на нужное устройство
        print("Инициализация генератора эмбеддингов завершена.")
        print(f"Содержимое папки temp_emb: {os.listdir('temp_emb') if os.path.exists('temp_emb') else 'Папка не существует'}.")

    def generate_embeddings_batch(self, texts: list) -> torch.Tensor:
        """
        Генерирует эмбеддинги для списка текстов с использованием BERT.
        """
        print(f"Начинаем генерацию эмбеддингов для {len(texts)} текстов.")
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)  # Средний эмбеддинг по всем токенам
        print(f"Генерация эмбеддингов завершена. Размер эмбеддинга: {embeddings.shape}")
        return embeddings

    def generate_embedding(self, text: str) -> torch.Tensor:
        """
        Генерирует эмбеддинг для одного текста.
        """
        embedding = self.generate_embeddings_batch([text])[0]
        print(f"Генерация эмбеддинга завершена. Размерность: {embedding.shape}")
        
        # Сохраняем эмбеддинг
        embedding_filepath = self.save_embedding(embedding)
        if embedding_filepath:
            print(f"Эмбеддинг успешно сохранён в файл: {embedding_filepath}")
        else:
            print("Ошибка при сохранении эмбеддинга.")
        return embedding

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
