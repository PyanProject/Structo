import torch
from embedding_generator import generate_embedding_from_text
from model_generator import generate_3d_scene_from_embedding, visualize_3d_scene

def main():
    # Выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Ввод текста для создания эмбеддинга и 3D модели
    text = input("Введите текст для создания эмбеддинга и 3D сцены: ")

    # Генерация эмбеддинга
    embedding = generate_embedding_from_text(text, device)
    print(f"Генерация эмбеддинга завершена. Размерность: {embedding.shape}")

    # Генерация 3D модели
    scene = generate_3d_scene_from_embedding(embedding.cpu().numpy())
    visualize_3d_scene(scene)

if __name__ == "__main__":
    main()
