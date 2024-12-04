import numpy as np
import open3d as o3d

def generate_3d_scene_from_embedding(embedding: np.ndarray) -> str:

    #Генерирует 3D сцену на основе эмбеддинга.

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

    # Сохранение сцены в файл .ply
    scene_filename = "generated_mesh.ply"
    o3d.io.write_triangle_mesh(scene_filename, mesh)
    print(f"3D модель сохранена в файл: {scene_filename}")

    return scene_filename
