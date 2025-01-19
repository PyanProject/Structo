'''
главный файл модели - он отвечает за запуск обучения всей модели на основе GAN. В конце работы
он выводит визуализацию 3D объекта на основе обученной модели для быстрой оценки результатов и
дебага.

'''

import torch
from embedding_generator import EmbeddingGenerator
from model_generator import generate_3d_scene_from_embedding
from dataset import ModelNet40Dataset
from gan_model import Generator, Discriminator, train_gan
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # инициализация устройства
    print(f"[MAIN] Используемое устройство: {device}")

    embedding_generator = EmbeddingGenerator(device, reduced_dim=512)

    dataset_generator = ModelNet40Dataset(root_dir="datasets/ModelNet40", split="train")
    dataloader = dataset_generator.generate_dataset()

    input_dim = 100
    output_dim = 3072
    generator = Generator(noise_dim=input_dim, embedding_dim=512, output_dim=output_dim).to(device)
    discriminator = Discriminator(data_dim=3072, embedding_dim=512).to(device)

    print("[MAIN] Начинаем тренировку GAN...")
    train_gan(generator, discriminator, dataloader, embedding_generator, epochs=10, lr=0.0002, device=device)

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("[MAIN] Модели GAN сохранены.")

    text = input("[MAIN] Введите текст для создания эмбеддинга и 3D сцены: ")
    print(f"[MAIN] Введенный текст: {text}")
    embedding = embedding_generator.generate_embedding(text).to(device)
    #embedding = embedding.squeeze()
    embedding = embedding_generator.generate_embedding(text).to(device)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)

    with torch.no_grad():
        noise = torch.randn(1, generator.noise_dim).to(device)
        # Verify dimensions before passing to generator
        print(f"Noise shape: {noise.shape}")
        print(f"Embedding shape: {embedding.shape}")
        generated_data = generator(noise, embedding).cpu().numpy().squeeze()

    scene_filename = generate_3d_scene_from_embedding(generated_data, text)
    print(f"[MAIN] 3D модель сохранена в файл: {scene_filename}")

    print("[MAIN] Открываем 3D модель для визуализации с помощью Open3D...")
    mesh = o3d.io.read_triangle_mesh(scene_filename)
    if not mesh.is_empty():
        o3d.visualization.draw_geometries([mesh])
    else:
        print("[MAIN] Ошибка при загрузке модели для визуализации.")

if __name__ == "__main__":
    main()
