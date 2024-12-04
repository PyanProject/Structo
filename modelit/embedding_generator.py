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
        print(f"Генерация эмбеддинга для текста: {text[:50]}...")
        return self.generate_embeddings_batch([text])[0]

    def save_embedding(self, embedding: torch.Tensor, output_dir: str = "temp_emb", unique_filename: bool = True) -> str:
        """
        Сохраняет эмбеддинг в файл .npy в папку temp_emb.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = "embedding.npy"
        if unique_filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embedding_{timestamp}.npy"

        filepath = os.path.join(output_dir, filename)

        try:
            np.save(filepath, embedding.cpu().numpy())  # Переводим на CPU перед сохранением
            print(f'Эмбеддинг успешно сохранён: {filepath}')
        except Exception as e:
            print(f'Ошибка при сохранении файла: {e}')
            return None

        return filepath
