import os
import torch
import numpy as np
import open3d as o3d
from dataset import ModelNet40Dataset, collate_fn
from embedding_generator import EmbeddingGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import spacy

def compute_dataset_embeddings(dataset, embedding_generator, device):
    """
    Вычисляет эмбеддинги для всех моделей в датасете и возвращает список кортежей:
    (file_path, embedding).
    """
    embeddings_list = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    # Обходим датасет
    for item in tqdm(dataloader, desc="Computing dataset embeddings"):
        if item is None:
            continue
        vertices, faces, class_name, file_path = item
        # Здесь для примера используем имя класса в качестве текстового описания
        text = class_name
        with torch.no_grad():
            emb = embedding_generator.generate_embedding(text).to(device)
        embeddings_list.append((file_path, emb.cpu()))
    return embeddings_list

def load_point_cloud(file_path):
    """
    Загрузка 3D модели из .off файла и возврат облака точек в виде numpy массива.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if lines[0].strip() != 'OFF':
        raise ValueError("Not a valid OFF file")
    n_verts, n_faces, _ = map(int, lines[1].strip().split())
    verts = [list(map(float, line.strip().split())) for line in lines[2:2 + n_verts]]
    return np.array(verts)

def retrieval_by_text(query_text, dataset_embeddings, embedding_generator, device):
    """
    Выполняет поиск ближайшей модели по текстовому запросу.
    Возвращает файл с моделью и его эмбеддинг, если найден.
    """
    with torch.no_grad():
        query_emb = embedding_generator.generate_embedding(query_text).to(device)
    query_emb = query_emb.cpu()
    
    best_score = float('inf')
    best_file = None
    # Простой евклид (L2) расстояние между эмбеддингами
    for file_path, emb in dataset_embeddings:
        distance = torch.norm(query_emb - emb)
        if distance < best_score:
            best_score = distance
            best_file = file_path
    return best_file, best_score

def generate_mesh_from_point_cloud(pts):
    """
    Из облака точек генерирует 3D меш с использованием Poisson reconstruction (с сглаживанием).
    pts: numpy массив точек (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))
    print("[RETRIEVAL] Performing Poisson surface reconstruction.")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RETRIEVAL] Using device: {device}")

    # Загрузка spaCy моделей
    try:
        print("[RETRIEVAL] Loading spaCy models...")
        spacy.load("ru_core_news_sm")
        spacy.load("en_core_web_sm")
        print("[RETRIEVAL] spaCy models loaded.")
    except Exception as e:
        print(f"[RETRIEVAL] Error loading spaCy models: {e}")
        exit(1)

    # Инициализация EmbeddingGenerator
    try:
        embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
        print("[RETRIEVAL] EmbeddingGenerator initialized.")
    except Exception as e:
        print(f"[RETRIEVAL] Error initializing EmbeddingGenerator: {e}")
        exit(1)

    # Загрузка датасета
    dataset_path = "datasets/CoolDataset"
    try:
        dataset = ModelNet40Dataset(root_dir=dataset_path, split="train")
        print("[RETRIEVAL] Dataset loaded.")
    except Exception as e:
        print(f"[RETRIEVAL] Error loading dataset: {e}")
        exit(1)
    
    # Предварительный расчет эмбеддингов датасета
    print("[RETRIEVAL] Computing dataset embeddings...")
    dataset_embeddings = compute_dataset_embeddings(dataset, embedding_generator, device)
    print("[RETRIEVAL] Done computing embeddings.")

    while True:
        query = input("[RETRIEVAL] Enter text query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("[RETRIEVAL] Exiting...")
            break
        best_file, score = retrieval_by_text(query, dataset_embeddings, embedding_generator, device)
        if best_file is None:
            print("[RETRIEVAL] No matching model found.")
            continue
        print(f"[RETRIEVAL] Best matching file: {best_file} (distance: {score:.4f})")
        pts = load_point_cloud(best_file)
        mesh = generate_mesh_from_point_cloud(pts)
        # Визуализация полученного меша
        o3d.visualization.draw_geometries([mesh], window_name="Retrieved 3D Model") 