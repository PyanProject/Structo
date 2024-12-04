import numpy as np
import open3d as o3d
import threading

def generate_3d_scene_from_embedding(embedding: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Генерирует 3D сцену на основе эмбеддинга.
    """
    # Генерация случайных точек 3D сцены из эмбеддинга (пример)
    num_points = 1000  # Количество точек
    points = np.random.rand(num_points, 3)  # Генерация случайных точек
    colors = np.random.rand(num_points, 3)  # Случайные цвета для точек

    # Создание объекта Open3D для визуализации
    pcd = o3d.geometry.PointCloud()

    # Установка координат точек
    pcd.points = o3d.utility.Vector3dVector(points)

    # Установка цветов точек
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_3d_scene_thread(pcd):
    """
    Запускает визуализацию 3D сцены в отдельном потоке.
    """
    o3d.visualization.draw_geometries([pcd])

def visualize_3d_scene(pcd):
    """
    Визуализирует 3D сцену в отдельном потоке.
    """
    # Визуализация в отдельном потоке
    visualization_thread = threading.Thread(target=visualize_3d_scene_thread, args=(pcd,))
    visualization_thread.start()
    visualization_thread.join()
