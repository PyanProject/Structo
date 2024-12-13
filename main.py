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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Используемое устройство: {device}")

    embedding_generator = EmbeddingGenerator(device, reduced_dim=512)

    dataset_generator = CustomDataset()
    samples = dataset_generator.generate_dataset()
    dataset = Temporary3DDataset(samples, embedding_generator)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    input_dim = 512
    output_dim = 512
    generator = Generator(input_dim=input_dim, output_dim=output_dim).to(device)
    discriminator = Discriminator(input_dim=output_dim).to(device)

    print("[MAIN] Начинаем тренировку GAN...")
    train_gan(generator, discriminator, dataloader, epochs=10, lr=0.0002, device=device)

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("[MAIN] Модели GAN сохранены.")

    text = input("[MAIN] Введите текст для создания эмбеддинга и 3D сцены: ")
    print(f"[MAIN] Введенный текст: {text}")
    embedding = embedding_generator.generate_embedding(text)
    print(f"[MAIN] Генерация эмбеддинга завершена. Размерность: {embedding.shape}")

    with torch.no_grad():
        noise = torch.randn(1, input_dim).to(device)
        generated_embedding = generator(noise).cpu().numpy().squeeze()

    scene_filename = generate_3d_scene_from_embedding(generated_embedding, text)
    print(f"[MAIN] 3D модель сохранена в файл: {scene_filename}")

    print("[MAIN] Открываем 3D модель для визуализации с помощью Open3D...")
    mesh = o3d.io.read_triangle_mesh(scene_filename)
    if not mesh.is_empty():
        o3d.visualization.draw_geometries([mesh])
    else:
        print("[MAIN] Ошибка при загрузке модели для визуализации.")

if __name__ == "__main__":
    main()
