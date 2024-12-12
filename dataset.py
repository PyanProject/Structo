# dataset.py

import os
import trimesh
from torch.utils.data import Dataset
import numpy as np

class CustomDataset:
    """
    Генерация собственного датасета с текстовыми описаниями и 3D моделями.
    """
    def __init__(self, output_dir="temp_dataset"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = [
            {"text": "Красная сфера с радиусом 1", "model": "sphere"},
            {"text": "Синий куб со стороной 2", "model": "cube"},
            {"text": "Зелёный конус с высотой 1.5", "model": "cone"},
            {"text": "Жёлтый цилиндр с радиусом 1 и высотой 2", "model": "cylinder"},
            {"text": "Фиолетовая пирамида с основанием 1 и высотой 2", "model": "pyramid"},
            {"text": "Оранжевый тор с major_radius 1 и minor_radius 0.3", "model": "torus"},
            {"text": "Белая цилиндрическая труба с длиной 3 и радиусом 0.5", "model": "cylindrical_tube"},
            {"text": "Чёрная сфера с радиусом 0.75 и текстурой", "model": "textured_sphere"},
            {"text": "Серая кубическая коробка с отверстиями", "model": "holed_cube"},
            {"text": "Розовый конус с двунаправленной высотой", "model": "double_cone"},
            # Добавьте больше образцов по необходимости
        ]
    
    def generate_dataset(self):
        samples = []
        for idx, sample in enumerate(self.data):
            model_type = sample["model"]
            if model_type == "sphere":
                mesh = trimesh.creation.icosphere(radius=1.0)
            elif model_type == "cube":
                mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
            elif model_type == "cone":
                mesh = trimesh.creation.cone(radius=1.0, height=1.5)
            elif model_type == "cylinder":
                mesh = trimesh.creation.cylinder(radius=1.0, height=2.0)
            elif model_type == "pyramid":
                mesh = self.create_pyramid(extents=[1.0, 1.0, 2.0])
            elif model_type == "torus":
                # Исправлено: передача major_radius и minor_radius
                mesh = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
            elif model_type == "cylindrical_tube":
                mesh = trimesh.creation.cylinder(radius=0.5, height=3.0)
            elif model_type == "textured_sphere":
                mesh = trimesh.creation.icosphere(radius=0.75)
                # Добавьте текстуру при необходимости
            elif model_type == "holed_cube":
                mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
                # Добавьте отверстия при необходимости
            elif model_type == "double_cone":
                mesh = trimesh.creation.cone(radius=1.0, height=3.0)
            else:
                print(f"Неизвестный тип модели: {model_type}. Пропускаем.")
                continue  # Пропустить неизвестные модели
            
            filepath = os.path.join(self.output_dir, f"sample_{idx}.ply")
            mesh.export(filepath)
            samples.append({"text": sample["text"], "filepath": filepath})
            print(f"Создан образец {idx}: {sample['text']} -> {filepath}")
        
        print(f"Создано {len(samples)} образцов в {self.output_dir}")
        return samples

    def create_pyramid(self, extents=[1.0, 1.0, 2.0]) -> trimesh.Trimesh:
        """
        Создаёт пирамиду с квадратным основанием.
        
        :param extents: Список из трех значений [ширина, глубина, высота]
        :return: Объект Trimesh представляющий пирамиду
        """
        width, depth, height = extents
        # Вершины пирамиды
        vertices = np.array([
            [0, 0, 0],  # Основание
            [width, 0, 0],
            [width, depth, 0],
            [0, depth, 0],
            [width / 2, depth / 2, height]  # Вершина
        ])
        
        # Грани пирамиды
        faces = np.array([
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [0, 1, 2],
            [0, 2, 3]
        ])
        
        pyramid = trimesh.Trimesh(vertices=vertices, faces=faces)
        return pyramid

class Temporary3DDataset(Dataset):
    """
    Класс для загрузки датасета с эмбеддингами для 3D моделей.
    """
    def __init__(self, samples, embedding_generator):
        self.samples = samples
        self.embedding_generator = embedding_generator

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        embedding = self.embedding_generator.generate_embedding(text).squeeze()  # Генерация эмбеддинга из текста
        filepath = sample["filepath"]
        return embedding, filepath
