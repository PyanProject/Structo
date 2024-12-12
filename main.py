# main.py

import torch
from embedding_generator import EmbeddingGenerator
from model_generator import generate_3d_scene_from_embedding
from dataset import CustomDataset, Temporary3DDataset
from gan_model import Generator, Discriminator, train_gan
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import os

def main():
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Инициализация генератора эмбеддингов
    embedding_generator = EmbeddingGenerator(device, reduced_dim=512)  # Используем 512-мерные эмбеддинги

    # Генерация датасета
    dataset_generator = CustomDataset()
    samples = dataset_generator.generate_dataset()
    dataset = Temporary3DDataset(samples, embedding_generator)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Инициализация GAN
    input_dim = 512  # Размерность эмбеддинга CLIP
    output_dim = 512
    generator = Generator(input_dim=input_dim, output_dim=output_dim).to(device)
    discriminator = Discriminator(input_dim=output_dim).to(device)

    # Тренировка GAN
    print("Начинаем тренировку GAN...")
    train_gan(generator, discriminator, dataloader, epochs=10, lr=0.0002, device=device)

    # Сохранение моделей GAN
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Модели GAN сохранены как 'generator.pth' и 'discriminator.pth'.")

    # Генерация 3D модели с использованием GAN
    print("Генерация 3D модели на основе текста...")
    text = input("Введите текст для создания эмбеддинга и 3D сцены: ")
    print(f"Введенный текст: {text}")

    # Генерация эмбеддинга
    embedding = embedding_generator.generate_embedding(text)
    print(f"Генерация эмбеддинга завершена. Размерность: {embedding.shape}")

    # Генерация модели через GAN
    with torch.no_grad():
        noise = torch.randn(1, input_dim).to(device)  # Случайный шум для генерации
        generated_embedding = generator(noise).cpu().numpy().squeeze()
    
    # Создание 3D модели
    scene_filename = generate_3d_scene_from_embedding(generated_embedding, text)
    print(f"3D модель сохранена в файл: {scene_filename}")

    # Открытие и визуализация модели с использованием Open3D
    print(f"Открываем 3D модель для визуализации с помощью Open3D...")
    mesh = o3d.io.read_triangle_mesh(scene_filename)
    if not mesh.is_empty():
        o3d.visualization.draw_geometries([mesh])
    else:
        print("Ошибка при загрузке модели для визуализации.")

if __name__ == "__main__":
    main()
