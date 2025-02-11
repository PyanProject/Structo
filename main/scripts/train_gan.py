#train_gan.py
import os
import torch
import spacy
from tqdm import tqdm

from utils.clip_embedding import EmbeddingGenerator
from utils.dataset_utils import ModelNet40Dataset, collate_fn
from utils.gan_models import Generator, Discriminator, train_gan
from torch.utils.data import DataLoader

def validate_dataset(dataset):
    valid_files = []
    print("[TRAIN] Проверка целостности файлов датасета...")
    for idx in tqdm(range(len(dataset.file_list)), desc="Проверка файлов"):
        try:
            item = dataset[idx]
            if item is not None:
                valid_files.append(dataset.file_list[idx])
        except Exception as e:
            print(f"\nФайл {dataset.file_list[idx]} повреждён: {e}")
    dataset.file_list = valid_files
    print(f"[TRAIN] Валидных файлов: {len(valid_files)}")
    if len(valid_files) == 0:
        raise ValueError("Датасет не содержит валидных файлов!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Используемое устройство: {device}")

    try:
        print("[TRAIN] Загрузка моделей spaCy...")
        spacy.load("ru_core_news_sm")
        spacy.load("en_core_web_sm")
        print("[TRAIN] Модели spaCy загружены.")
    except Exception as e:
        print(f"[TRAIN] Ошибка: {e}")
        print("Установите модели: `python -m spacy download ru_core_news_sm en_core_web_sm`")
        return

    try:
        embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
        print("[TRAIN] EmbeddingGenerator готов.")
    except Exception as e:
        print(f"[TRAIN] Ошибка инициализации EmbeddingGenerator: {e}")
        return

    try:
        dataset = ModelNet40Dataset(root_dir="datasets/CoolDataset", split="train")
        print("[TRAIN] Датасет загружен.")
    except Exception as e:
        print(f"[TRAIN] Ошибка загрузки датасета: {e}")
        return

    try:
        validate_dataset(dataset)
    except ValueError as e:
        print(f"[TRAIN] Критическая ошибка: {e}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    print(f"[TRAIN] DataLoader создан. Количество батчей: {len(dataloader)}")

    if len(dataloader) == 0:
        print("[TRAIN] Нет данных для обучения!")
        return

    try:
        generator = Generator(noise_dim=100, embedding_dim=512).to(device)
        discriminator = Discriminator(data_dim=12288, embedding_dim=512).to(device)
        print("[TRAIN] GAN инициализирован.")
    except Exception as e:
        print(f"[TRAIN] Ошибка инициализации GAN: {e}")
        return

    print("[TRAIN] Запуск обучения...")
    try:
        train_gan(generator, discriminator, dataloader, embedding_generator, epochs=3, lr=0.0001, device=device)
        print("[TRAIN] Обучение завершено.")
    except Exception as e:
        print(f"[TRAIN] Ошибка обучения: {e}")
        return

    os.makedirs('models/checkpoints', exist_ok=True)
    torch.save(generator.state_dict(), 'models/checkpoints/generator.pth')
    torch.save(discriminator.state_dict(), 'models/checkpoints/discriminator.pth')
    print("[TRAIN] Модели сохранены.")

if __name__ == "__main__":
    main()
