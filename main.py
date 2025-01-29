'''
главный файл модели - он отвечает за запуск обучения всей модели на основе GAN.
'''

import torch
import trimesh
from embedding_generator import EmbeddingGenerator
from model_generator import generate_3d_scene_from_embedding
from dataset import ModelNet40Dataset, collate_fn
from gan_model import Generator, Discriminator, train_gan
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import os
import spacy
from tqdm import tqdm  # Для отображения прогресса

def validate_dataset(dataset):
    '''Проверка целостности файлов датасета с прогресс-баром'''
    valid_files = []
    print("[MAIN] Проверка целостности файлов датасета...")
    for idx in tqdm(range(len(dataset.file_list)), desc="Проверка файлов"):
        try:
            item = dataset[idx]
            if item is not None:
                valid_files.append(dataset.file_list[idx])
        except Exception as e:
            print(f"\nФайл {dataset.file_list[idx]} повреждён: {e}")
    dataset.file_list = valid_files
    print(f"[MAIN] Валидных файлов: {len(valid_files)}/{len(dataset.file_list)}")
    if len(valid_files) == 0:
        raise ValueError("Датасет не содержит валидных файлов!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"[MAIN] Используемое устройство: {device}")

    # Проверка моделей spaCy
    try:
        print("[MAIN] Загрузка моделей spaCy...")
        spacy.load("ru_core_news_sm")
        spacy.load("en_core_web_sm")
        print("[MAIN] Модели spaCy загружены.")
    except Exception as e:
        print(f"[MAIN] Ошибка: {e}")
        print("Установите модели: `python -m spacy download ru_core_news_sm en_core_web_sm`")
        return

    # Инициализация компонентов
    print("[MAIN] Инициализация EmbeddingGenerator...")
    try:
        embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
        print("[MAIN] EmbeddingGenerator готов.")
    except Exception as e:
        print(f"[MAIN] Ошибка инициализации EmbeddingGenerator: {e}")
        return

    # Загрузка датасета
    print("[MAIN] Загрузка датасета...")
    try:
        dataset_generator = ModelNet40Dataset(root_dir="datasets/CoolDataset", split="train")
        print("[MAIN] Датасет загружен.")
    except Exception as e:
        print(f"[MAIN] Ошибка загрузки датасета: {e}")
        return

    # Валидация датасета
    try:
        validate_dataset(dataset_generator)
    except ValueError as e:
        print(f"[MAIN] Критическая ошибка: {e}")
        return

    # DataLoader
    print("[MAIN] Создание DataLoader...")
    dataloader = DataLoader(
        dataset_generator,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    print(f"[MAIN] DataLoader создан. Количество батчей: {len(dataloader)}")

    # Проверка наличия данных
    if len(dataloader) == 0:
        print("[MAIN] Нет данных для обучения!")
        return

    # Инициализация GAN
    print("[MAIN] Инициализация GAN...")
    try:
        generator = Generator(noise_dim=100, embedding_dim=512).to(device)
        discriminator = Discriminator(data_dim=3072, embedding_dim=512).to(device)
        print("[MAIN] GAN инициализирован.")
    except Exception as e:
        print(f"[MAIN] Ошибка инициализации GAN: {e}")
        return

    # Обучение
    print("[MAIN] Запуск обучения...")
    try:
        train_gan(generator, discriminator, dataloader, embedding_generator, epochs=1, lr=0.0001, device=device)
        print("[MAIN] Обучение завершено.")
    except Exception as e:
        print(f"[MAIN] Ошибка обучения: {e}")
        return

    # Сохранение моделей
    os.makedirs('models', exist_ok=True)
    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')
    print("[MAIN] Модели сохранены.")

    while True:
        text = input("[MAIN] Введите текст для генерации 3D-модели (или 'exit' для выхода): ")
        if text.lower() in ["exit", "quit"]:
            print("[MAIN] Завершение работы...")
            break

        try:
            embedding = embedding_generator.generate_embedding(text).to(device)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            with torch.no_grad():
                noise = torch.randn(1, generator.noise_dim).to(device)
                generated_data = generator(noise, embedding).cpu().detach().numpy().squeeze()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(generated_data)
                print("[DEBUG] Визуализация сгенерированных точек...")
                o3d.visualization.draw_geometries([pcd], window_name="Сгенерированные точки")
                print("[DEBUG] Визуализация сохраненного мэша...")
                #mesh = trimesh.load(saved_mesh)
                #mesh.show()
                

            scene_filename = generate_3d_scene_from_embedding(generated_data, text)
            print(f"[MAIN] Модель сохранена: {scene_filename}")
        except Exception as e:
            print(f"[MAIN] Ошибка генерации: {e}")

if __name__ == "__main__":
    main()