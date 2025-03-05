import torch
import torch.nn as nn
import numpy as np
import trimesh
import os
import json
from typing import Optional, Dict, List, Tuple, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Настройка логгера.
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу для записи логов
        
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def save_config(config: Dict, path: str):
    """
    Сохранение конфигурации в JSON файл.
    
    Args:
        config: Словарь с конфигурацией
        path: Путь для сохранения
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
        
def load_config(path: str) -> Dict:
    """
    Загрузка конфигурации из JSON файла.
    
    Args:
        path: Путь к файлу конфигурации
        
    Returns:
        Словарь с конфигурацией
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def count_parameters(model: nn.Module) -> int:
    """
    Подсчет количества параметров модели.
    
    Args:
        model: Модель PyTorch
        
    Returns:
        Количество обучаемых параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_voxels(
    voxels: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    save_path: Optional[str] = None
):
    """
    Визуализация воксельной сетки.
    
    Args:
        voxels: Воксельная сетка
        threshold: Порог для бинаризации
        save_path: Путь для сохранения визуализации
    """
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()
        
    # Бинаризация
    voxels = voxels > threshold
    
    # Создаем фигуру
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Получаем координаты занятых вокселей
    x, y, z = np.where(voxels)
    
    # Отрисовываем
    ax.scatter(x, y, z, c='b', marker='s')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()

def create_gif(
    frames: List[Image.Image],
    save_path: str,
    duration: int = 100
):
    """
    Создание GIF анимации из списка изображений.
    
    Args:
        frames: Список изображений
        save_path: Путь для сохранения GIF
        duration: Длительность каждого кадра в миллисекундах
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Нормализация меша (центрирование и масштабирование).
    
    Args:
        mesh: Исходный меш
        
    Returns:
        Нормализованный меш
    """
    # Центрирование
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices = mesh.vertices - centroid
    
    # Масштабирование
    scale = np.abs(mesh.vertices).max()
    mesh.vertices = mesh.vertices / scale
    
    return mesh

def setup_device(
    use_cuda: bool = True,
    memory_fraction: float = 1.0
) -> torch.device:
    """
    Настройка устройства для вычислений.
    
    Args:
        use_cuda: Использовать ли CUDA
        memory_fraction: Доля памяти GPU для использования
        
    Returns:
        Устройство PyTorch
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        
        # Настройка памяти CUDA
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        
    return device

def create_timestamp_dir(base_dir: str) -> str:
    """
    Создание директории с временной меткой.
    
    Args:
        base_dir: Базовая директория
        
    Returns:
        Путь к созданной директории
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(base_dir, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

class AverageMeter:
    """Класс для подсчета и хранения среднего значения метрики."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Сброс значений."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        """
        Обновление значений.
        
        Args:
            val: Новое значение
            n: Вес значения
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 