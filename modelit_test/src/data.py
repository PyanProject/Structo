#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для работы с данными, включая загрузку и обработку 3D моделей из Objaverse.
"""

import os
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import trimesh
    from trimesh.voxel import creation as voxel_creation
except ImportError:
    logging.warning("Trimesh не установлен. Некоторые функции будут недоступны.")

try:
    import mcubes
except ImportError:
    logging.warning("PyMCubes не установлен. Некоторые функции будут недоступны.")

# Настройка логгера
logger = logging.getLogger(__name__)

class MeshDataset(Dataset):
    """
    Датасет для работы с 3D мешами из Objaverse или других источников.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        voxel_size: int = 64,
        num_views: int = 4,
        transform: Optional[Callable] = None,
        is_objaverse: bool = True,
        cache_path: Optional[str] = None,
        load_in_memory: bool = False,
    ):
        """
        Инициализация датасета.
        
        Args:
            data_dir: Директория с данными.
            split: Разбиение данных ('train', 'val', 'test').
            voxel_size: Размер воксельной сетки.
            num_views: Количество ракурсов для рендеринга.
            transform: Функция трансформации данных.
            is_objaverse: Указывает, что данные в формате Objaverse.
            cache_path: Путь для кэширования обработанных данных.
            load_in_memory: Загружать все данные в память.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.voxel_size = voxel_size
        self.num_views = num_views
        self.transform = transform
        self.is_objaverse = is_objaverse
        self.cache_path = Path(cache_path) if cache_path else None
        self.load_in_memory = load_in_memory
        
        # Загрузка метаданных
        self.metadata = self.load_metadata()
        
        # Создание индекса сэмплов для выбранного разбиения
        self.samples = self.create_index()
        
        # Кэшированные данные, если включена загрузка в память
        self.cached_data = {}
        
        # Предзагрузка данных, если включена загрузка в память
        if self.load_in_memory:
            self._preload_data()
            
        logger.info(f"Создан датасет из {len(self.samples)} образцов для разбиения '{split}'")
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Загрузка метаданных.
        
        Returns:
            Словарь с метаданными.
        """
        metadata_path = self.data_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        logger.info(f"Загружены метаданные для {len(metadata)} объектов")
        return metadata
    
    def create_index(self) -> List[str]:
        """
        Создание индекса сэмплов для выбранного разбиения.
        
        Returns:
            Список ID объектов для выбранного разбиения.
        """
        split_file = self.data_dir / f"{self.split}.txt"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Файл разбиения не найден: {split_file}")
        
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        
        # Проверяем, что все ID есть в метаданных
        valid_ids = [obj_id for obj_id in ids if obj_id in self.metadata]
        
        if len(valid_ids) < len(ids):
            logger.warning(f"Найдено {len(valid_ids)} из {len(ids)} объектов в метаданных")
        
        return valid_ids
    
    def _preload_data(self):
        """
        Предзагрузка данных в память.
        """
        logger.info(f"Предзагрузка {len(self.samples)} объектов в память...")
        start_time = time.time()
        
        for i, obj_id in enumerate(self.samples):
            if i % 100 == 0:
                logger.info(f"Загружено {i}/{len(self.samples)} объектов...")
            
            try:
                # Загружаем данные для объекта
                voxels = self.voxelize_mesh(obj_id)
                views = self.render_views(obj_id, voxels)
                
                # Кэшируем данные
                self.cached_data[obj_id] = {
                    "voxels": voxels,
                    "views": views,
                    "caption": self.metadata[obj_id].get("caption", ""),
                }
            except Exception as e:
                logger.error(f"Ошибка при загрузке объекта {obj_id}: {e}")
        
        end_time = time.time()
        logger.info(f"Предзагрузка завершена за {end_time - start_time:.2f} секунд")
    
    def load_mesh(self, obj_id: str) -> Optional[trimesh.Trimesh]:
        """
        Загрузка 3D меша по ID.
        
        Args:
            obj_id: ID объекта.
            
        Returns:
            Загруженный меш или None, если произошла ошибка.
        """
        try:
            if self.is_objaverse:
                # Путь к файлу меша в формате Objaverse
                obj_path = self.data_dir / self.metadata[obj_id]["path"] / "model.obj"
                
                # Если файла нет, пытаемся найти другие форматы
                if not obj_path.exists():
                    obj_dir = self.data_dir / self.metadata[obj_id]["path"]
                    mesh_files = list(obj_dir.glob("*.obj")) + list(obj_dir.glob("*.ply")) + list(obj_dir.glob("*.stl"))
                    
                    if not mesh_files:
                        logger.warning(f"Меш-файл не найден для объекта {obj_id}")
                        return None
                    
                    obj_path = mesh_files[0]
            else:
                # Путь к файлу меша в обычном формате
                obj_path = self.data_dir / self.metadata[obj_id]["path"]
            
            # Загружаем меш
            mesh = trimesh.load(obj_path)
            
            # Нормализация меша
            mesh = self.normalize_mesh(mesh)
            
            return mesh
        except Exception as e:
            logger.error(f"Ошибка при загрузке меша {obj_id}: {e}")
            return None
    
    def normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Нормализация меша (центрирование и масштабирование).
        
        Args:
            mesh: Исходный меш.
            
        Returns:
            Нормализованный меш.
        """
        # Центрирование
        centroid = mesh.centroid
        mesh.vertices -= centroid
        
        # Масштабирование
        scale = 1.0 / max(mesh.extents)
        mesh.vertices *= scale
        
        return mesh
    
    def voxelize_mesh(self, obj_id: str) -> np.ndarray:
        """
        Преобразование меша в воксельное представление.
        
        Args:
            obj_id: ID объекта.
            
        Returns:
            Воксельное представление меша.
        """
        # Проверяем, есть ли данные в кэше
        if self.cached_data and obj_id in self.cached_data:
            return self.cached_data[obj_id]["voxels"]
        
        # Проверяем, есть ли кэшированные данные на диске
        if self.cache_path:
            cache_file = self.cache_path / f"{obj_id}_voxel_{self.voxel_size}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        voxels = pickle.load(f)
                    return voxels
                except Exception as e:
                    logger.warning(f"Ошибка при загрузке кэша для {obj_id}: {e}")
        
        # Загружаем меш
        mesh = self.load_mesh(obj_id)
        
        if mesh is None:
            # Если не удалось загрузить меш, возвращаем пустой массив
            return np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.float32)
        
        try:
            # Преобразуем меш в воксели
            voxels = voxel_creation.voxelize(
                mesh, 
                pitch=2.0/self.voxel_size, 
                method="ray"
            ).sparse_indices
            
            # Преобразуем индексы воксельной сетки в бинарную 3D сетку
            voxel_grid = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.float32)
            
            if len(voxels) > 0:
                # Смещаем и масштабируем индексы к диапазону [0, voxel_size-1]
                vox_idx = np.floor((voxels + 1.0) * (self.voxel_size / 2)).astype(int)
                
                # Фильтруем индексы, выходящие за пределы сетки
                valid_idx = np.all((vox_idx >= 0) & (vox_idx < self.voxel_size), axis=1)
                vox_idx = vox_idx[valid_idx]
                
                # Заполняем вокселы
                if len(vox_idx) > 0:
                    voxel_grid[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = 1.0
            
            # Кэшируем результат, если включено кэширование
            if self.cache_path:
                os.makedirs(self.cache_path, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(voxel_grid, f)
            
            return voxel_grid
        except Exception as e:
            logger.error(f"Ошибка при воксилизации меша {obj_id}: {e}")
            return np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.float32)
    
    def render_views(self, obj_id: str, voxels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Рендеринг нескольких ракурсов объекта.
        
        Args:
            obj_id: ID объекта.
            voxels: Воксельное представление объекта (если уже вычислено).
            
        Returns:
            Массив с рендерингами ракурсов [num_views, H, W].
        """
        # Проверяем, есть ли данные в кэше
        if self.cached_data and obj_id in self.cached_data:
            return self.cached_data[obj_id]["views"]
        
        # Проверяем, есть ли кэшированные данные на диске
        if self.cache_path:
            cache_file = self.cache_path / f"{obj_id}_views_{self.num_views}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        views = pickle.load(f)
                    return views
                except Exception as e:
                    logger.warning(f"Ошибка при загрузке кэша для {obj_id}: {e}")
        
        # Если вокселы не предоставлены, вычисляем их
        if voxels is None:
            voxels = self.voxelize_mesh(obj_id)
        
        # Размер результирующих изображений
        view_size = 64
        
        # Создаем массив для хранения ракурсов
        views = np.zeros((self.num_views, view_size, view_size), dtype=np.float32)
        
        try:
            # Генерируем ракурсы (простой метод - проекции)
            for i in range(self.num_views):
                # Угол поворота
                angle = i * (360 / self.num_views)
                
                # Проекция (упрощенно - суммирование вдоль оси, определяемой углом)
                if angle % 90 == 0:
                    axis = (angle // 90) % 3
                    view = np.sum(voxels, axis=axis)
                else:
                    # Для других углов используем более сложную проекцию (не реализовано в этом примере)
                    view = np.sum(voxels, axis=0)
                
                # Масштабируем к размеру view_size
                from scipy.ndimage import zoom
                view = zoom(view, view_size / view.shape[0], order=1)
                
                # Нормализуем значения
                if view.max() > 0:
                    view = view / view.max()
                
                views[i] = view
            
            # Кэшируем результат, если включено кэширование
            if self.cache_path:
                os.makedirs(self.cache_path, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(views, f)
            
            return views
        except Exception as e:
            logger.error(f"Ошибка при рендеринге ракурсов для {obj_id}: {e}")
            return np.zeros((self.num_views, view_size, view_size), dtype=np.float32)
    
    def __len__(self) -> int:
        """
        Количество сэмплов в датасете.
        
        Returns:
            Количество сэмплов.
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Получение сэмпла по индексу.
        
        Args:
            idx: Индекс сэмпла.
            
        Returns:
            Словарь с данными сэмпла.
        """
        try:
            # Получаем ID объекта
            obj_id = self.samples[idx]
            
            # Проверяем, есть ли данные в кэше
            if self.cached_data and obj_id in self.cached_data:
                voxels = self.cached_data[obj_id]["voxels"]
                views = self.cached_data[obj_id]["views"]
                caption = self.cached_data[obj_id]["caption"]
            else:
                # Получаем воксельное представление
                voxels = self.voxelize_mesh(obj_id)
                
                # Рендерим ракурсы
                views = self.render_views(obj_id, voxels)
                
                # Получаем описание
                caption = self.metadata[obj_id].get("caption", "")
            
            # Конвертируем в тензоры
            voxels_tensor = torch.from_numpy(voxels).float()
            views_tensor = torch.from_numpy(views).float()
            
            # Применяем трансформации, если они заданы
            if self.transform:
                voxels_tensor = self.transform(voxels_tensor)
            
            return {
                "id": obj_id,
                "voxels": voxels_tensor,
                "views": views_tensor,
                "caption": caption
            }
        except Exception as e:
            logger.error(f"Ошибка при получении сэмпла {idx}: {e}")
            
            # Возвращаем пустой сэмпл в случае ошибки
            empty_voxels = torch.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=torch.float32)
            empty_views = torch.zeros((self.num_views, 64, 64), dtype=torch.float32)
            
            return {
                "id": "error",
                "voxels": empty_voxels,
                "views": empty_views,
                "caption": "Error loading sample"
            }

def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    voxel_size: int = 64,
    transform: Optional[Callable] = None,
    is_objaverse: bool = True,
    cache_path: Optional[str] = None,
    load_in_memory: bool = False,
) -> DataLoader:
    """
    Создание DataLoader'а для загрузки данных.
    
    Args:
        data_dir: Директория с данными.
        split: Разбиение данных ('train', 'val', 'test').
        batch_size: Размер батча.
        shuffle: Перемешивать ли данные.
        num_workers: Количество рабочих процессов для загрузки данных.
        voxel_size: Размер воксельной сетки.
        transform: Функция трансформации данных.
        is_objaverse: Указывает, что данные в формате Objaverse.
        cache_path: Путь для кэширования обработанных данных.
        load_in_memory: Загружать все данные в память.
        
    Returns:
        DataLoader для загрузки данных.
    """
    dataset = MeshDataset(
        data_dir=data_dir,
        split=split,
        voxel_size=voxel_size,
        transform=transform,
        is_objaverse=is_objaverse,
        cache_path=cache_path,
        load_in_memory=load_in_memory,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader 