import torch
from embedding_generator import EmbeddingGenerator
from model_generator import generate_3d_scene_from_embedding
from dataset import TemporaryDataset
from gan_model import Generator, Discriminator, train_gan
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Temporary3DDataset(Dataset):
    """
    Класс для загрузки временного 3D датасета.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        embedding = torch.tensor(np.random.rand(128), dtype=torch.float32)  # Генерация случайного эмбеддинга для обучения
        filepath = sample["filepath"]
        return embedding, filepath


def main():
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Генерация временного датасета
    dataset_generator = TemporaryDataset()
    samples = dataset_generator.generate_dataset(num_samples=50)
    dataset = Temporary3DDataset(samples)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Инициализация генератора эмбеддингов
    embedding_generator = EmbeddingGenerator(device)

    # Тренировка GAN
    print("Начинаем тренировку GAN...")
    input_dim = 128  # Размерность эмбеддинга
    output_dim = 128  # Выходная размерность (примерно соответствует числу вершин)
    generator = Generator(input_dim=input_dim, output_dim=output_dim).to(device)
    discriminator = Discriminator(input_dim=output_dim).to(device)

    train_gan(generator, discriminator, dataloader, epochs=10, lr=0.0002, device=device)

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


if __name__ == "__main__":
    main()
