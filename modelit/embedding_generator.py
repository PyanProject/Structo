import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import datetime
import os

def generate_embedding_from_text(text: str, device: torch.device) -> torch.Tensor:
    """
    Генерирует эмбеддинг для текста с использованием BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)  # Переносим модель на нужное устройство

    # Токенизация текста
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Получаем эмбеддинги
    with torch.no_grad():
        outputs = model(**inputs)

    # Средний эмбеддинг
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding

def save_embedding(embedding: torch.Tensor, output_dir: str = "temp_emb", unique_filename: bool = True) -> str:
    """
    Сохраняет эмбеддинг в файл .npy в папку temp_emb.
    """
    # Убедимся, что папка существует
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
