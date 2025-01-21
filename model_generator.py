'''
этот файл раньше помогал дегенерировать простые модели, а сейчас его нужно переделать под датасет

'''

import numpy as np
import trimesh
import os
import hashlib
import open3d as o3d


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

def generate_3d_scene_from_embedding(generated_data, text, output_dir="models"):
    print("[MODEL GEN] Генерация сцены...")
    manage_model_files(output_dir)
    scene_filename = generate_unique_filename(text, output_dir)
    
    # Проверка данных
    assert generated_data.ndim == 2 and generated_data.shape[1] == 3, "Generated data must be of shape (N, 3)"
    if np.isnan(generated_data).any() or np.isinf(generated_data).any():
        print("[MODEL GEN] Generated data contains NaNs or Infs. Cannot create mesh.")
        return None

    # Создание облака точек
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(generated_data)

    # Расчет нормалей
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))  # Ориентация по оси Z
    
    # Реконструкция меша с использованием Ball-Pivoting
    radii = [0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    
    # Сохранение
    o3d.io.write_triangle_mesh(scene_filename, mesh)
    print(f"[MODEL GEN] Модель сохранена: {scene_filename}")
    return scene_filename