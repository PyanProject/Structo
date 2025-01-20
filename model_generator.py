import numpy as np
import open3d as o3d
import os
import hashlib

def generate_unique_filename(text: str, output_dir: str) -> str:
    hash_object = hashlib.md5(text.encode())
    filename = f"model_{hash_object.hexdigest()[:8]}.ply"
    return os.path.join(output_dir, filename)

def manage_model_files(output_dir: str, max_files: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(output_dir) if f.endswith('.ply')]
    if len(files) > max_files:
        oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
        os.remove(os.path.join(output_dir, oldest_file))
        print(f"[MODEL GEN] Удалён старый файл: {oldest_file}")

def generate_3d_scene_from_embedding(generated_data: np.ndarray, text: str, output_dir: str = "models") -> str:
    print("[MODEL GEN] Генерация сцены...")
    
    manage_model_files(output_dir)
    scene_filename = generate_unique_filename(text, output_dir)
    
    # Ensure data integrity
    assert generated_data.ndim == 2 and generated_data.shape[1] == 3, "Generated data must be of shape (N, 3)"
    if np.isnan(generated_data).any() or np.isinf(generated_data).any():
        print("[MODEL GEN] Generated data contains NaNs or Infs. Cannot create mesh.")
        return None

    # Create a point cloud from generated_data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(generated_data)
    
    # Downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    
    # Orient normals
    pcd.orient_normals_to_align_with_direction()
    
    # Scale the point cloud
    pcd.scale(1.0 / np.std(np.asarray(pcd.points)), center=pcd.get_center())
    
    # Optional: Visualize the point cloud with normals
    # o3d.visualization.draw_geometries([pcd])

    # Optional: Apply Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    o3d.io.write_triangle_mesh(scene_filename, mesh)
    print(f"[MODEL GEN] Модель сохранена: {scene_filename}")
    return scene_filename