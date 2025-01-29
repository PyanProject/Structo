import os
import json
import shutil
import trimesh
import numpy as np
from sklearn.model_selection import train_test_split


def load_meshes(file_paths):
    meshes = []
    for path in file_paths:
        try:
            mesh = trimesh.load(path)
            meshes.append(mesh)
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
    return meshes

def move_files(files, dest_dir):
    """Перемещает файлы в целевую директорию, создавая структуру классов"""
    for file_path in files:
        class_name = os.path.basename(os.path.dirname(file_path))
        class_dest_dir = os.path.join(dest_dir, class_name)
        os.makedirs(class_dest_dir, exist_ok=True)
        try:
            shutil.move(file_path, class_dest_dir)
        except Exception as e:
            print(f"Ошибка при перемещении {file_path}: {e}")


# Путь к папке с датасетом
dataset_path = r"C:\Users\Tu-tu-ru\Documents\GitHub\modelit\datasets\CoolDataset"

# Собираем все файлы .off
all_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".off"):
            all_files.append(os.path.join(root, file))

print(f"Всего файлов: {len(all_files)}")

labels = [os.path.basename(os.path.dirname(f)) for f in all_files]

# Разделение (например, 80% тренировочных, 20% тестовых)
train_files, test_files = train_test_split(
    all_files,
    test_size=0.2,          # Размер тестовой выборки
    random_state=42,        # Для воспроизводимости
    stratify=labels,
    shuffle=True            # Перемешать данные перед разделением
)

print(f"Обучающих файлов: {len(train_files)}")
print(f"Тестовых файлов: {len(test_files)}")

train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

print("Начинаем перемещение тренировочных данных...")
move_files(train_files, train_dir)

print("\nНачинаем перемещение тестовых данных...")
move_files(test_files, test_dir)


print("\nГотово! Результирующая структура:")
print(f"Train: {len(train_files)} файлов")
print(f"Test: {len(test_files)} файлов")
