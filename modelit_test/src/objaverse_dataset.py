#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import json
import logging
import requests
import numpy as np
import psutil
import torch
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Класс для отслеживания и управления памятью"""
    
    @staticmethod
    def get_memory_usage():
        """Возвращает текущее использование памяти в МБ"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        return memory_usage_mb
    
    @staticmethod
    def check_memory(threshold_mb=10000):  # 10GB порог для систем с 16GB RAM
        """Проверяет использование памяти и выполняет сборку мусора если превышен порог"""
        current_usage = MemoryManager.get_memory_usage()
        if current_usage > threshold_mb:
            logger.warning(f"Высокое использование памяти: {current_usage:.2f}MB. Выполняется сборка мусора...")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            new_usage = MemoryManager.get_memory_usage()
            logger.info(f"Использование памяти после сборки: {new_usage:.2f}MB (освобождено {current_usage - new_usage:.2f}MB)")

def download_object_data(uid: str) -> Dict[str, Any]:
    """Загружает данные объекта по его UID из Objaverse API"""
    try:
        response = requests.get(f"https://api.objaverse.com/api/v1/object/{uid}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных объекта {uid}: {e}")
        return {"uid": uid, "description": "", "category": ""}

def download_point_cloud(uid: str) -> np.ndarray:
    """Загружает облако точек для объекта по его UID"""
    try:
        url = f"https://objaverse.allenai.org/point_clouds/{uid}.npy"
        response = requests.get(url)
        response.raise_for_status()
        point_cloud = np.load(BytesIO(response.content))
        return point_cloud
    except Exception as e:
        logger.error(f"Ошибка при загрузке облака точек для {uid}: {e}")
        # Возвращаем случайное облако точек
        return np.random.normal(0, 1, (4096, 3)).astype(np.float32)

def download_image(uid: str) -> Image.Image:
    """Загружает изображение для объекта по его UID"""
    try:
        url = f"https://objaverse.allenai.org/renders/{uid}/0.png"
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Ошибка при загрузке изображения для {uid}: {e}")
        # Возвращаем пустое изображение
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return img

def generate_test_data(output_dir: str, num_objects: int = 1000, split_ratio: float = 0.9,
                      max_points: int = 4096):
    """
    Генерирует тестовые данные для отладки
    
    Args:
        output_dir: Директория для сохранения данных
        num_objects: Количество объектов для генерации
        split_ratio: Соотношение обучающих/проверочных данных
        max_points: Максимальное количество точек в облаке
    """
    logger.info(f"Генерация {num_objects} тестовых объектов...")
    
    output_dir = Path(output_dir)
    
    # Создаем директории для данных
    images_dir = output_dir / "images"
    points_dir = output_dir / "points"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем список объектов
    objects = []
    categories = ["стол", "стул", "диван", "лампа", "шкаф", "кровать", "ваза", "компьютер", "телефон", "автомобиль"]
    
    for i in tqdm(range(num_objects), desc="Генерация тестовых данных"):
        uid = f"test_object_{i:06d}"
        category = categories[i % len(categories)]
        
        # Генерируем случайное изображение
        img = Image.new("RGB", (224, 224), color=(
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256)
        ))
        img_path = images_dir / f"{uid}.jpg"
        img.save(img_path)
        
        # Генерируем случайное облако точек
        points = np.random.normal(0, 0.5, (max_points, 3)).astype(np.float32)
        # Нормализуем облако точек в диапазон [-1, 1]
        points = points / np.max(np.abs(points)) if np.max(np.abs(points)) > 0 else points
        points_path = points_dir / f"{uid}.npy"
        np.save(points_path, points)
        
        # Создаем метаданные
        object_data = {
            "uid": uid,
            "text_description": f"3D модель {category} в стиле минимализм",
            "category": category,
            "split": "train" if i < num_objects * split_ratio else "val"
        }
        
        objects.append(object_data)
        
        # Периодически проверяем память
        if i % 100 == 0:
            MemoryManager.check_memory()
    
    # Сохраняем метаданные
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(objects, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Тестовые данные сгенерированы и сохранены в {output_dir}")
    logger.info(f"Обучающих объектов: {len([o for o in objects if o['split'] == 'train'])}")
    logger.info(f"Проверочных объектов: {len([o for o in objects if o['split'] == 'val'])}")

def prepare_objaverse_data(output_dir: str, num_objects: int = 1000, split_ratio: float = 0.9,
                          max_points: int = 4096, batch_size: int = 5):
    """
    Подготавливает данные из Objaverse для обучения
    
    Args:
        output_dir: Директория для сохранения данных
        num_objects: Количество объектов для загрузки
        split_ratio: Соотношение обучающих/проверочных данных
        max_points: Максимальное количество точек в облаке
        batch_size: Размер батча для загрузки
    """
    logger.info("Начало подготовки данных Objaverse")
    
    try:
        # Проверяем доступность API Objaverse
        test_response = requests.head("https://api.objaverse.com/api/v1/list_objects")
        if test_response.status_code != 200:
            logger.warning(f"API Objaverse недоступен (код {test_response.status_code}). Использую тестовые данные.")
            generate_test_data(output_dir, num_objects, split_ratio, max_points)
            return
        
        output_dir = Path(output_dir)
        
        # Создаем директории для данных
        images_dir = output_dir / "images"
        points_dir = output_dir / "points"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        points_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем список объектов из Objaverse
        logger.info("Получение списка объектов из Objaverse...")
        response = requests.get("https://api.objaverse.com/api/v1/list_objects")
        response.raise_for_status()
        all_objects = response.json()
        
        # Ограничиваем количество объектов
        selected_objects = all_objects[:num_objects]
        logger.info(f"Выбрано {len(selected_objects)} объектов из {len(all_objects)}")
        
        # Подготавливаем данные по батчам
        objects_data = []
        
        for i in tqdm(range(0, len(selected_objects), batch_size), desc="Загрузка объектов"):
            batch = selected_objects[i:i+batch_size]
            
            for uid in batch:
                # Загрузка метаданных объекта
                object_data = download_object_data(uid)
                
                # Загрузка изображения
                img_path = images_dir / f"{uid}.jpg"
                if not img_path.exists():
                    try:
                        img = download_image(uid)
                        img.save(img_path)
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении изображения для {uid}: {e}")
                
                # Загрузка облака точек
                points_path = points_dir / f"{uid}.npy"
                if not points_path.exists():
                    try:
                        points = download_point_cloud(uid)
                        # Ограничиваем количество точек
                        if points.shape[0] > max_points:
                            indices = np.random.choice(points.shape[0], max_points, replace=False)
                            points = points[indices]
                        elif points.shape[0] < max_points:
                            # Добавляем недостающие точки как копии существующих
                            pad_indices = np.random.choice(points.shape[0], max_points - points.shape[0], replace=True)
                            points = np.vstack([points, points[pad_indices]])
                        
                        np.save(points_path, points)
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении облака точек для {uid}: {e}")
                
                # Определяем split (train или val)
                object_data["split"] = "train" if i < len(selected_objects) * split_ratio else "val"
                objects_data.append(object_data)
            
            # Проверка использования памяти
            MemoryManager.check_memory()
        
        # Сохраняем метаданные
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(objects_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Данные подготовлены и сохранены в {output_dir}")
        logger.info(f"Обучающих объектов: {len([o for o in objects_data if o['split'] == 'train'])}")
        logger.info(f"Проверочных объектов: {len([o for o in objects_data if o['split'] == 'val'])}")
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке данных: {e}")
        logger.info("Переключение на генерацию тестовых данных...")
        generate_test_data(output_dir, num_objects, split_ratio, max_points)

class ObjaverseDataset(Dataset):
    """Датасет для работы с данными Objaverse"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, max_points: int = 4096):
        """
        Инициализирует датасет
        
        Args:
            data_dir: Директория с данными
            split: 'train' или 'val'
            transform: трансформации для изображений
            max_points: максимальное количество точек в облаке
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_points = max_points
        
        # Загружаем метаданные
        metadata_path = self.data_dir / "metadata.json"
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке метаданных: {e}")
            metadata = []
        
        # Фильтруем по split
        self.objects = [obj for obj in metadata if obj.get('split') == split]
        
        if not self.objects:
            logger.warning(f"Нет объектов для split '{split}'")
        
        logger.info(f"Загружено {len(self.objects)} объектов для split '{split}'")
    
    def _load_and_process_points(self, points_path: Path) -> torch.Tensor:
        """
        Загружает и обрабатывает облако точек
        
        Args:
            points_path: путь к файлу .npy с облаком точек
            
        Returns:
            Тензор с нормализованным облаком точек [N, 3]
        """
        try:
            # Используем memory mapping для экономии памяти
            points = np.load(points_path, mmap_mode='r')
            
            # Копируем только нужные данные
            if points.shape[0] > self.max_points:
                indices = np.random.choice(points.shape[0], self.max_points, replace=False)
                points = points[indices].copy()
            elif points.shape[0] < self.max_points:
                # Если точек меньше, дублируем существующие
                actual_points = points.copy()
                num_to_sample = self.max_points - points.shape[0]
                indices = np.random.choice(points.shape[0], num_to_sample, replace=True)
                sampled_points = points[indices].copy()
                points = np.vstack([actual_points, sampled_points])
            else:
                points = points.copy()
            
            # Нормализуем в диапазон [-1, 1]
            max_abs = np.max(np.abs(points))
            if max_abs > 0:
                points = points / max_abs
            
            # Преобразуем в тензор
            points_tensor = torch.from_numpy(points).float()
            
            return points_tensor
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке облака точек {points_path}: {e}")
            # Возвращаем случайное облако точек
            return torch.randn(self.max_points, 3)
    
    def _load_and_process_image(self, image_path: Path) -> torch.Tensor:
        """
        Загружает и обрабатывает изображение
        
        Args:
            image_path: путь к файлу изображения
            
        Returns:
            Тензор с нормализованным изображением [3, H, W]
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Базовая обработка: изменение размера и нормализация
                image = image.resize((224, 224))
                image = np.array(image) / 255.0
                image = torch.from_numpy(image).float().permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            return image
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения {image_path}: {e}")
            # Возвращаем пустой тензор
            return torch.zeros(3, 224, 224)
    
    def __len__(self) -> int:
        """Возвращает количество объектов в датасете"""
        return len(self.objects)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Возвращает объект по индексу
        
        Args:
            idx: индекс объекта
            
        Returns:
            Словарь с данными объекта:
            - 'uid': уникальный идентификатор
            - 'text': текстовое описание
            - 'points': облако точек [N, 3]
            - 'image': изображение [3, H, W]
            - 'category': категория объекта
        """
        obj = self.objects[idx]
        uid = obj['uid']
        
        # Пути к файлам
        points_path = self.data_dir / "points" / f"{uid}.npy"
        image_path = self.data_dir / "images" / f"{uid}.jpg"
        
        # Загрузка данных
        points = self._load_and_process_points(points_path)
        image = self._load_and_process_image(image_path)
        
        # Формируем результат
        result = {
            'uid': uid,
            'text': obj.get('text_description', ''),
            'points': points,
            'image': image,
            'category': obj.get('category', '')
        }
        
        return result

def get_objaverse_dataloader(data_dir: str, batch_size: int, split: str = 'train',
                            num_workers: int = 4, max_points: int = 4096) -> DataLoader:
    """
    Создаёт DataLoader для датасета Objaverse
    
    Args:
        data_dir: директория с данными
        batch_size: размер батча
        split: 'train' или 'val'
        num_workers: количество рабочих процессов для загрузки данных
        max_points: максимальное количество точек в облаке
        
    Returns:
        DataLoader для указанного split
    """
    dataset = ObjaverseDataset(
        data_dir=data_dir,
        split=split,
        transform=None,  # Можно добавить трансформации для аугментации
        max_points=max_points
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    logger.info(f"Создан DataLoader для split '{split}' с {len(dataset)} объектами, batch_size={batch_size}")
    
    return dataloader 