'''
этот файл раньше помогал дегенерировать простые модели, а сейчас его нужно переделать под датасет

'''

import numpy as np
import trimesh
import os
import hashlib
import open3d as o3d
import matplotlib.pyplot as plt


# при генерации модели она сохраняется и ей присваивается уникальное имя. очень важная ф-ция
def generate_unique_filename(text: str, output_dir: str) -> str:
    hash_object = hashlib.md5(text.encode())
    filename = f"model_{hash_object.hexdigest()[:8]}.ply"
    return os.path.join(output_dir, filename)

# не дает файлам переполниться, ф-ция-мусорщик
def manage_model_files(output_dir: str, max_files: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(output_dir) if f.endswith('.ply')]
    if len(files) > max_files:
        oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
        os.remove(os.path.join(output_dir, oldest_file))
        print(f"[MODEL GEN] Удалён старый файл: {oldest_file}")

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

import trimesh
import numpy as np
import matplotlib.pyplot as plt

def generate_3d_scene_from_embedding(embedding, text, faces, output_dir="models"):
    try:
        # Преобразование эмбеддинга в вершины
        vertices = embedding.reshape(-1, 3)
        print(f"[DEBUG] Вершины: {vertices[:5]}")  # Вывод первых 5 вершин для отладки

        # Использование реальных данных для граней
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f"[DEBUG] Грани: {faces[:5]}")  # Вывод первых 5 граней для отладки

        # Альтернативная визуализация с использованием matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, cmap='viridis', edgecolor='none')
        plt.show()
        print("[DEBUG] Визуализация выполнена")

        # Сохранение модели
        scene_filename = generate_unique_filename(text, output_dir)
        mesh.export(scene_filename)
        manage_model_files(output_dir)
        print(f"[MODEL GEN] 3D модель сохранена в файл: {scene_filename}")
        return scene_filename
    except Exception as e:
        print(f"[ERROR] Ошибка при генерации 3D сцены: {e}")
        return None
    

def generate_faces(vertices):
    # Пример генерации граней для простого случая
    # Здесь предполагается, что vertices представляет собой сетку точек
    # Для более сложных случаев используйте алгоритмы триангуляции
    faces = []
    for i in range(len(vertices) - 1):
        if i % 2 == 0:
            faces.append([i, i + 1, (i + 2) % len(vertices)])
    return np.array(faces)