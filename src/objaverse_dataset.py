import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import gc
from typing import Dict, Any
import psutil
import logging
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import requests
from io import BytesIO
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    @staticmethod
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    @staticmethod
    def check_memory(threshold_mb=10000):  # Уменьшаем порог до 10GB для систем с 16GB RAM
        memory_used = MemoryManager.get_memory_usage()
        if memory_used > threshold_mb:
            logger.warning(f"High memory usage detected: {memory_used:.2f}MB")
            gc.collect()
            torch.cuda.empty_cache()

def download_object_data(uid: str) -> Dict[str, Any]:
    """Загрузка данных объекта через API Objaverse"""
    base_url = "https://api.objaverse.com/api/v1"
    response = requests.get(f"{base_url}/object/{uid}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get object data: {response.status_code}")

def download_point_cloud(uid: str) -> np.ndarray:
    """Загрузка облака точек через API Objaverse"""
    base_url = "https://api.objaverse.com/api/v1"
    response = requests.get(f"{base_url}/object/{uid}/point_cloud")
    if response.status_code == 200:
        points = np.frombuffer(response.content, dtype=np.float32).reshape(-1, 3)
        return points
    else:
        raise Exception(f"Failed to get point cloud: {response.status_code}")

def download_image(uid: str) -> Image.Image:
    """Загрузка изображения через API Objaverse"""
    base_url = "https://api.objaverse.com/api/v1"
    response = requests.get(f"{base_url}/object/{uid}/image")
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Failed to get image: {response.status_code}")

def generate_test_data(output_dir: str, num_objects: int = 1000, split_ratio: float = 0.9,
                      max_points: int = 4096):
    """
    Генерация тестовых данных для отладки
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "points").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {num_objects} test objects")
    
    # Генерируем тестовые объекты
    objects = []
    for i in range(num_objects):
        uid = str(uuid.uuid4())
        objects.append({
            'uid': uid,
            'text': f'Test object {i}',
            'category': f'category_{i % 5}'
        })
    
    # Разделяем на train и val
    train_size = int(len(objects) * split_ratio)
    train_objects = objects[:train_size]
    val_objects = objects[train_size:]
    
    splits = {
        'train': train_objects,
        'val': val_objects
    }
    
    for split, split_objects in splits.items():
        metadata = {}
        for obj in split_objects:
            uid = obj['uid']
            
            # Генерируем тестовое изображение
            image = Image.new('RGB', (224, 224), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            image_path = output_dir / split / "images" / f"{uid}.jpg"
            image.save(image_path, quality=90, optimize=True)
            
            # Генерируем тестовое облако точек - координаты трехмерных точек
            # Важно: сохраняем в формате [max_points, 3], но при загрузке транспонируем
            points = np.random.randn(max_points, 3).astype(np.float32)
            
            # Нормализация точек
            points_min = points.min(axis=0, keepdims=True)
            points_max = points.max(axis=0, keepdims=True)
            points = 2 * (points - points_min) / (points_max - points_min + 1e-8) - 1
            
            points_path = output_dir / split / "points" / f"{uid}.npy"
            np.save(points_path, points)
            
            metadata[uid] = {
                'text_description': obj['text'],
                'category': obj['category'],
            }
        
        # Сохраняем метаданные
        metadata_path = output_dir / f"{split}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    logger.info("Test data generation completed")

def prepare_objaverse_data(output_dir: str, num_objects: int = 1000, split_ratio: float = 0.9,
                          max_points: int = 4096, batch_size: int = 5):
    """
    Подготовка данных из Objaverse или генерация тестовых данных
    """
    try:
        # Для отладки используем тестовые данные
        generate_test_data(output_dir, num_objects, split_ratio, max_points)
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise

class ObjaverseDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', transform=None, max_points: int = 4096):
        """
        Оптимизированная инициализация датасета Objaverse для систем с ограниченной памятью
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_points = max_points
        self.cache: Dict[str, Any] = {}
        self.max_cache_size = 50  # Уменьшаем размер кэша для экономии памяти
        
        metadata_path = self.data_dir / f"{split}_metadata.json"
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
            
        self.uids = list(self.metadata.keys())
        logger.info(f"Loaded {len(self.uids)} samples for {split} split")
    
    def _load_and_process_points(self, points_path: Path) -> torch.Tensor:
        """Загрузка и обработка облака точек с оптимизацией памяти"""
        try:
            points = np.load(points_path, mmap_mode='r')  # Используем memory mapping для больших файлов
            
            # Нормализация в float32 для экономии памяти
            points = points.astype(np.float32)
            
            # Убедимся, что у нас правильное количество точек
            if len(points) > self.max_points:
                indices = np.random.choice(len(points), self.max_points, replace=False)
                points = points[indices]
            elif len(points) < self.max_points:
                # Дополняем случайными точками до нужного размера
                extra_points = np.random.randn(self.max_points - len(points), 3).astype(np.float32)
                points = np.concatenate([points, extra_points], axis=0)
            
            # Нормализация
            points_min = points.min(axis=0, keepdims=True)
            points_max = points.max(axis=0, keepdims=True)
            points = 2 * (points - points_min) / (points_max - points_min + 1e-8) - 1
            
            # Преобразуем в формат [3 x max_points]
            points = points.transpose()
            
            return torch.from_numpy(points).float()
        except Exception as e:
            logger.error(f"Error loading points from {points_path}: {e}")
            # Создаем случайное облако точек с правильной размерностью
            return torch.randn(3, self.max_points, dtype=torch.float32)  # [channels x points]
    
    def _load_and_process_image(self, image_path: Path) -> torch.Tensor:
        """Загрузка и обработка изображения с оптимизацией памяти"""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = np.array(image, dtype=np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)
            return image
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            return torch.zeros(3, 224, 224, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.uids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        MemoryManager.check_memory()
        
        uid = self.uids[idx]
        
        # Проверяем кэш
        if uid in self.cache:
            return self.cache[uid]
        
        item_data = self.metadata[uid]
        
        # Загрузка данных
        points_path = self.data_dir / self.split / "points" / f"{uid}.npy"
        image_path = self.data_dir / self.split / "images" / f"{uid}.jpg"
        
        points = self._load_and_process_points(points_path)  # [3 x max_points]
        image = self._load_and_process_image(image_path)  # [3 x H x W]
        text = item_data.get('text_description', '')
        
        batch = {
            'uid': uid,
            'points': points,  # [3 x max_points]
            'image': image,  # [3 x H x W]
            'text': text
        }
        
        # Обновляем кэш, удаляя старые элементы при необходимости
        if len(self.cache) >= self.max_cache_size:
            remove_key = next(iter(self.cache))
            del self.cache[remove_key]
        
        self.cache[uid] = batch
        return batch

def get_objaverse_dataloader(data_dir: str, batch_size: int, split: str = 'train',
                            num_workers: int = 4, max_points: int = 4096) -> DataLoader:
    """
    Оптимизированный DataLoader для систем с ограниченной памятью
    """
    try:
        dataset = ObjaverseDataset(data_dir, split=split, max_points=max_points)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        )
    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise 