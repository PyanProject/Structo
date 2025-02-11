import numpy as np
import trimesh
import os
import hashlib
import open3d as o3d

def generate_unique_filename(text: str, output_dir: str) -> str:
    hash_object = hashlib.md5(text.encode())
    filename = f"model_{hash_object.hexdigest()[:8]}.obj"
    return os.path.join(output_dir, filename)

def manage_model_files(output_dir: str, max_files: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(output_dir) if f.endswith('.obj')]
    if len(files) > max_files:
        oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
        os.remove(os.path.join(output_dir, oldest_file))
        print(f"[MESH GEN] Удалён старый файл: {oldest_file}")

def generate_3d_scene_from_embedding(generated_data, text, output_dir="models/generated"):
    print("[MESH GEN] Generating 3D scene...")
    os.makedirs(output_dir, exist_ok=True)
    manage_model_files(output_dir)
    scene_filename = generate_unique_filename(text, output_dir)

    # Если входной массив имеет размерность (1, N, 3), убираем батчевое измерение
    if generated_data.ndim == 3 and generated_data.shape[0] == 1:
        generated_data = np.squeeze(generated_data, axis=0)
        print(f"[MESH GEN] Squeezed generated data to shape: {generated_data.shape}")

    # Если текст содержит "sphere" или "сфера", генерируем сферу
    if "sphere" in text.lower() or "сфера" in text.lower():
        print("[MESH GEN] Detected 'sphere' or 'сфера' in text. Generating sphere mesh.")
        try:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(scene_filename, mesh)
            print(f"[MESH GEN] Sphere mesh saved: {scene_filename}")
            return scene_filename
        except Exception as e:
            print(f"[MESH GEN] Error generating sphere mesh: {e}")
            return None

    # Проверяем форму входных данных
    if not (generated_data.ndim == 2 and generated_data.shape[1] == 3):
        print("[MESH GEN] Error: Generated data must be of shape (N, 3).")
        return None
    if np.isnan(generated_data).any() or np.isinf(generated_data).any():
        print("[MESH GEN] Generated data contains NaNs or Infs. Cannot create mesh.")
        return None

    try:
        print("[MESH GEN] Creating point cloud from generated data...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(generated_data)
        print(f"[MESH GEN] Point cloud contains {len(pcd.points)} points.")
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        
        print("[MESH GEN] Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("[MESH GEN] Orienting normals...")
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))
        
        print("[MESH GEN] Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        densities = np.asarray(densities)
        print(f"[MESH GEN] Poisson reconstruction completed. Mesh has {len(mesh.triangles)} triangles.")
        
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        if len(mesh.triangles) < 50:
            print("[MESH GEN] Sparse mesh detected after Poisson reconstruction. Falling back to convex hull.")
            mesh, _ = pcd.compute_convex_hull()
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            mesh.compute_vertex_normals()
        else:
            mesh = mesh.filter_smooth_simple(number_of_iterations=3)
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            mesh.compute_vertex_normals()
        
        print("[MESH GEN] Saving mesh to file...")
        o3d.io.write_triangle_mesh(scene_filename, mesh)
        print(f"[MESH GEN] Mesh saved: {scene_filename}")
        return scene_filename
    except Exception as e:
        print(f"[MESH GEN] Error during mesh generation: {e}")
        return None
