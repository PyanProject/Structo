import torch
from embedding_generator import EmbeddingGenerator
from model_generator import generate_3d_scene_from_embedding

def main():
    # Выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Создаем объект для генерации эмбеддингов
    embedding_generator = EmbeddingGenerator(device)

    # Ввод текста для создания эмбеддинга и 3D модели
    text = input("Введите текст для создания эмбеддинга и 3D сцены: ")
    print(f"Введенный текст: {text}")

    # Генерация эмбеддинга
    embedding = embedding_generator.generate_embedding(text)
    print(f"Генерация эмбеддинга завершена. Размерность: {embedding.shape}")

    # Генерация 3D модели с полигонами
    scene_filename = generate_3d_scene_from_embedding(embedding.cpu().numpy(), text)
    print(f"3D модель сохранена в файл: {scene_filename}")

if __name__ == "__main__":
    main()
