"""
utils_mesh.py
--------------
Этот модуль объединяет функции для работы с 3D-моделями:
  • Загрузка мешей и облаков точек.
  • Перемещение файлов в структуре датасета и его организацию.
  • Генерация 3D меша из облака точек.
  • Вычисление эмбеддингов датасета и поиск моделей по текстовому запросу.
"""

# Общие импорты для работы с файловой системой, массивами и моделями
import os
import json
import shutil
import trimesh
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import open3d as o3d
from torch.utils.data import DataLoader
from tqdm import tqdm
import spacy

# Импорты для работы с датасетом и генерацией эмбеддингов
from dataset import ModelNet40Dataset, collate_fn
from embedding_generator import EmbeddingGenerator


# ============================================================================
# Функции для загрузки мешей и работы с файлами
# ============================================================================

def load_meshes(file_paths):
    """
    Загружает меши из списка файлов с использованием библиотеки trimesh.
    """
    meshes = []
    for path in file_paths:
        try:
            mesh = trimesh.load(path)
            meshes.append(mesh)
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
    return meshes

def move_files(files, dest_dir):
    """
    Перемещает файлы в целевую директорию, создавая структуру классов.
    """
    for file_path in files:
        class_name = os.path.basename(os.path.dirname(file_path))
        class_dest_dir = os.path.join(dest_dir, class_name)
        os.makedirs(class_dest_dir, exist_ok=True)
        try:
            shutil.move(file_path, class_dest_dir)
        except Exception as e:
            print(f"Ошибка при перемещении {file_path}: {e}")

def organize_dataset(dataset_path, train_dir=None, test_dir=None, test_size=0.2, random_state=42):
    """
    Собирает все файлы .off из dataset_path, разделяет их на тренировочные и тестовые, 
    перемещает их в указанные директории (если заданы) и выводит итоговую структуру.
    Если train_dir или test_dir не заданы, создаются в dataset_path.
    """
    # Собираем все .off файлы
    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".off"):
                all_files.append(os.path.join(root, file))
    
    print(f"Всего файлов: {len(all_files)}")
    
    labels = [os.path.basename(os.path.dirname(f)) for f in all_files]
    
    # Разделение файлов
    train_files, test_files = train_test_split(
        all_files,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
        shuffle=True
    )
    
    print(f"Обучающих файлов: {len(train_files)}")
    print(f"Тестовых файлов: {len(test_files)}")
    
    # Определяем директории для перемещения
    if train_dir is None:
        train_dir = os.path.join(dataset_path, "train")
    if test_dir is None:
        test_dir = os.path.join(dataset_path, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print("Начинаем перемещение тренировочных данных...")
    move_files(train_files, train_dir)
    
    print("\nНачинаем перемещение тестовых данных...")
    move_files(test_files, test_dir)
    
    print("\nГотово! Результирующая структура:")
    print(f"Train: {len(train_files)} файлов")
    print(f"Test: {len(test_files)} файлов")


# ============================================================================
# Функции для работы с облаками точек и генерации мешей
# ============================================================================

def load_point_cloud(file_path):
    """
    Загружает 3D модель из .off файла и возвращает облако точек в виде numpy массива.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if lines[0].strip() != 'OFF':
        raise ValueError("Not a valid OFF file")
    n_verts, n_faces, _ = map(int, lines[1].strip().split())
    verts = [list(map(float, line.strip().split())) for line in lines[2:2 + n_verts]]
    return np.array(verts)

def generate_mesh_from_point_cloud(pts):
    """
    Из облака точек генерирует 3D меш с использованием Poisson reconstruction (с сглаживанием).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))
    print("[UTILS_MESH] Performing Poisson surface reconstruction.")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh


# ============================================================================
# Функции для вычисления эмбеддингов датасета и поиска по тексту
# ============================================================================

def compute_dataset_embeddings(dataset, embedding_generator, device):
    """
    Вычисляет эмбеддинги для всех моделей в датасете и возвращает список кортежей:
    (file_path, embedding).
    """
    embeddings_list = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    for item in tqdm(dataloader, desc="Computing dataset embeddings"):
        if item is None:
            continue
        vertices, faces, class_name, file_path = item
        # Используем имя класса как текстовое описание
        text = class_name
        with torch.no_grad():
            emb = embedding_generator.generate_embedding(text).to(device)
        embeddings_list.append((file_path, emb.cpu()))
    return embeddings_list

def retrieval_by_text(query_text, dataset_embeddings, embedding_generator, device):
    """
    Выполняет поиск ближайшей модели по текстовому запросу.
    Возвращает путь к файлу модели и расстояние (метрика L2 между эмбеддингами).
    """
    with torch.no_grad():
        query_emb = embedding_generator.generate_embedding(query_text).to(device)
    query_emb = query_emb.cpu()
    
    best_score = float('inf')
    best_file = None
    for file_path, emb in dataset_embeddings:
        distance = torch.norm(query_emb - emb)
        if distance < best_score:
            best_score = distance
            best_file = file_path
    return best_file, best_score


# ============================================================================
# Пример использования (можно запускать для тестирования функционала модуля)
# ============================================================================

if __name__ == "__main__":
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UTILS_MESH] Using device: {device}")
    
    # Если требуется организовать датасет, можно раскомментировать следующую строку:
    # dataset_path = r"C:\Users\Tu-tu-ru\Documents\GitHub\modelit\datasets\CoolDataset"
    # organize_dataset(dataset_path)
    
    # Загрузка spaCy моделей
    try:
        print("[UTILS_MESH] Loading spaCy models...")
        spacy.load("ru_core_news_sm")
        spacy.load("en_core_web_sm")
        print("[UTILS_MESH] spaCy models loaded.")
    except Exception as e:
        print(f"[UTILS_MESH] Error loading spaCy models: {e}")
        exit(1)
    
    # Инициализация генератора эмбеддингов
    try:
        embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
        print("[UTILS_MESH] EmbeddingGenerator initialized.")
    except Exception as e:
        print(f"[UTILS_MESH] Error initializing EmbeddingGenerator: {e}")
        exit(1)
    
    # Загрузка датасета
    dataset_path = "datasets/CoolDataset"
    try:
        dataset = ModelNet40Dataset(root_dir=dataset_path, split="train")
        print("[UTILS_MESH] Dataset loaded.")
    except Exception as e:
        print(f"[UTILS_MESH] Error loading dataset: {e}")
        exit(1)
    
    # Вычисление эмбеддингов датасета
    print("[UTILS_MESH] Computing dataset embeddings...")
    dataset_embeddings = compute_dataset_embeddings(dataset, embedding_generator, device)
    print("[UTILS_MESH] Done computing embeddings.")
    
    # Тестовый цикл для поиска модели по текстовому запросу
    while True:
        query = input("[UTILS_MESH] Enter text query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("[UTILS_MESH] Exiting...")
            break
        best_file, score = retrieval_by_text(query, dataset_embeddings, embedding_generator, device)
        if best_file is None:
            print("[UTILS_MESH] No matching model found.")
            continue
        print(f"[UTILS_MESH] Best matching file: {best_file} (distance: {score:.4f})")
        pts = load_point_cloud(best_file)
        mesh = generate_mesh_from_point_cloud(pts)
        o3d.visualization.draw_geometries([mesh], window_name="Retrieved 3D Model")
