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
    print("[MODEL GEN] Generating 3D scene...")
    manage_model_files(output_dir)
    scene_filename = generate_unique_filename(text, output_dir)
    
    # Если текст содержит 'sphere', генерируем сферу напрямую
    if "sphere" in text.lower():
        print("[MODEL GEN] Detected 'sphere' in text. Generating sphere mesh.")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(scene_filename, mesh)
        print(f"[MODEL GEN] Sphere mesh saved: {scene_filename}")
        return scene_filename

    # Проверка данных
    assert generated_data.ndim == 2 and generated_data.shape[1] == 3, "Generated data must be of shape (N, 3)"
    if np.isnan(generated_data).any() or np.isinf(generated_data).any():
        print("[MODEL GEN] Generated data contains NaNs or Infs. Cannot create mesh.")
        return None

    # Создание облака точек
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(generated_data)
    # Устанавливаем единый цвет точек
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Расчет нормалей
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))
    
    # Используем Poisson surface reconstruction для генерации меша
    print("[MODEL GEN] Performing Poisson surface reconstruction.")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Если число треугольников слишком маленькое, значит Poisson реконструкция не смогла корректно восстановить форму,
    # поэтому переходим на вычисление выпуклой оболочки для получения замкнутой поверхности.
    if len(mesh.triangles) < 50:
         print("[MODEL GEN] Poisson reconstruction resulted in a sparse mesh. Falling back to convex hull.")
         mesh, _ = pcd.compute_convex_hull()
         mesh.paint_uniform_color([0.8, 0.8, 0.8])
         mesh.compute_vertex_normals()
    else:
         # Сглаживаем меш и устанавливаем единый цвет
         mesh = mesh.filter_smooth_simple(number_of_iterations=3)
         mesh.paint_uniform_color([0.8, 0.8, 0.8])
         mesh.compute_vertex_normals()
    
    # Save the mesh
    o3d.io.write_triangle_mesh(scene_filename, mesh)
    print(f"[MODEL GEN] Mesh saved: {scene_filename}")
    return scene_filename