# model_generator.py

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import hashlib

def generate_unique_filename(text: str, output_dir: str) -> str:
    """Создаёт уникальное имя файла на основе хэша текстового запроса."""
    hash_object = hashlib.md5(text.encode())
    filename = f"model_{hash_object.hexdigest()[:8]}.ply"
    return os.path.join(output_dir, filename)

def manage_model_files(output_dir: str, max_files: int = 10):
    """Удаляет старые файлы, если количество файлов в папке превышает max_files."""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(output_dir) if f.endswith('.ply')]
    if len(files) > max_files:
        oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
        os.remove(os.path.join(output_dir, oldest_file))
        print(f"Удалён старый файл: {oldest_file}")

def visualize_with_matplotlib(mesh):
    """Визуализация 3D модели с использованием matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Получаем вершины и грани
    vertices = mesh.vertices
    faces = mesh.faces

    # Создаем коллекцию полигонов
    poly3d = [[vertices[vert_idx] for vert_idx in face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, alpha=0.5, edgecolor='k'))

    # Устанавливаем границы осей
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    plt.show()

def normalize(value, min_val, max_val):
    """Нормализует значение в диапазон [0, 1]."""
    return (value - min_val) / (max_val - min_val)

def generate_3d_scene_from_embedding(embedding: np.ndarray, text: str, output_dir: str = "models") -> str:
    """Генерирует 3D сцену на основе эмбеддинга и сохраняет её в уникальный файл."""
    print("Начинаем генерацию 3D сцены из эмбеддинга.")
    
    # Нормализация эмбеддинга
    embedding_normalized = (embedding - embedding.mean()) / (embedding.std() + 1e-8)

    # Преобразуем эмбеддинг в параметры
    shape_param = embedding_normalized[0]  # Для выбора формы
    size_param = embedding_normalized[1]  # Для размера
    color_param = embedding_normalized[2:5]  # Для цвета

    # Нормализация параметров
    shape_param = normalize(shape_param, -1, 1)
    size_param = np.clip(size_param * 2, 0.5, 5.0)  # Размер от 0.5 до 5.0
    color = np.clip((color_param + 1) / 2, 0, 1)  # Цвет в диапазоне [0, 1]
    
    # Выбор формы на основе эмбеддинга
    if shape_param < 0.33:
        shape = "sphere"
    elif shape_param < 0.66:
        shape = "cube"
    else:
        shape = "cone"  # Изменено с "pyramid" на "cone" для совместимости с Trimesh
    
    # Генерация 3D объекта
    if shape == "sphere":
        mesh = trimesh.creation.icosphere(radius=size_param)
    elif shape == "cube":
        mesh = trimesh.creation.box(extents=[size_param, size_param, size_param])
    else:
        mesh = trimesh.creation.cone(radius=size_param, height=size_param * 1.5)

    # Устанавливаем цвет для модели
    mesh.visual.vertex_colors = (color * 255).astype(np.uint8)

    # Сохранение сцены в уникальный файл
    manage_model_files(output_dir)  # Проверяем и удаляем старые файлы, если нужно
    scene_filename = generate_unique_filename(text, output_dir)
    mesh.export(scene_filename)
    print(f"3D модель сохранена в файл: {scene_filename}")

    return scene_filename
