import numpy as np
import open3d as o3d
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

def generate_3d_scene_from_embedding(embedding: np.ndarray, text: str, output_dir: str = "models") -> str:
    """Генерирует 3D сцену на основе эмбеддинга и сохраняет её в уникальный файл."""
    print("Начинаем генерацию 3D сцены из эмбеддинга.")
    
    num_points = 1000  # Количество вершин
    points = np.random.rand(num_points, 3)  # Случайные вершины

    # Генерация случайных треугольников для создания 3D меша
    triangles = np.random.randint(0, num_points, size=(500, 3))  # Случайные индексы для треугольников

    # Создание объекта TriangleMesh в Open3D
    mesh = o3d.geometry.TriangleMesh()

    # Установка вершин
    mesh.vertices = o3d.utility.Vector3dVector(points)

    # Установка граней
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Генерация случайных цветов для вершин
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.random.rand(num_points, 3))

    # Визуализация
    print("Визуализация 3D модели...")
    o3d.visualization.draw_geometries([mesh])

    # Сохранение сцены в уникальный файл
    manage_model_files(output_dir)  # Проверяем и удаляем старые файлы, если нужно
    scene_filename = generate_unique_filename(text, output_dir)
    o3d.io.write_triangle_mesh(scene_filename, mesh)

    return scene_filename
