#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image
import trimesh


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Настройка логирования.
    
    Args:
        level: Уровень логирования.
        log_file: Путь к файлу для сохранения логов.
        
    Returns:
        Настроенный объект логгера.
    """
    logger = logging.getLogger("modelit")
    logger.setLevel(level)
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Добавляем обработчик для записи в файл, если указан путь
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Сохранение конфигурации в JSON файл.
    
    Args:
        config: Словарь с конфигурацией.
        path: Путь для сохранения конфигурации.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def load_config(path: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из JSON файла.
    
    Args:
        path: Путь к файлу конфигурации.
        
    Returns:
        Словарь с конфигурацией.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Подсчет количества обучаемых параметров в модели.
    
    Args:
        model: PyTorch модель.
        
    Returns:
        Количество обучаемых параметров.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_voxels(
    voxels: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Визуализация воксельной сетки.
    
    Args:
        voxels: Воксельная сетка формы [depth, height, width].
        threshold: Порог для отображения вокселей.
        figsize: Размер фигуры (ширина, высота).
        save_path: Путь для сохранения визуализации.
    """
    # Конвертируем в numpy, если необходимо
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()
    
    # Бинаризуем воксели по порогу
    binary_voxels = voxels > threshold
    
    # Визуализируем
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Получаем координаты активных вокселей
    z, y, x = np.where(binary_voxels)
    
    # Рисуем вокселы
    ax.scatter(x, y, z, marker='o', alpha=0.7)
    
    # Устанавливаем ограничения осей
    ax.set_xlim(0, voxels.shape[2])
    ax.set_ylim(0, voxels.shape[1])
    ax.set_zlim(0, voxels.shape[0])
    
    # Устанавливаем подписи осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Сохраняем, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def create_gif(
    images: List[Image.Image],
    output_path: str,
    duration: int = 100
) -> None:
    """
    Создание GIF-анимации из серии изображений.
    
    Args:
        images: Список изображений.
        output_path: Путь для сохранения GIF-файла.
        duration: Длительность каждого кадра в миллисекундах.
    """
    # Сохраняем анимацию
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Нормализация меша (центрирование и масштабирование).
    
    Args:
        mesh: Входной меш.
        
    Returns:
        Нормализованный меш.
    """
    # Создаем копию меша
    normalized_mesh = mesh.copy()
    
    # Центрируем меш
    center = (normalized_mesh.bounds[0] + normalized_mesh.bounds[1]) / 2
    normalized_mesh.vertices -= center
    
    # Масштабируем меш
    scale = np.max(normalized_mesh.bounds[1] - normalized_mesh.bounds[0])
    normalized_mesh.vertices /= (scale / 2.0)
    
    return normalized_mesh


def setup_device(cuda: bool = True) -> torch.device:
    """
    Настройка устройства для вычислений (CPU/GPU).
    
    Args:
        cuda: Флаг использования CUDA, если доступно.
        
    Returns:
        Устройство для вычислений.
    """
    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    
    return device


def create_timestamp_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Создание директории с временной меткой.
    
    Args:
        base_dir: Базовая директория.
        prefix: Префикс для названия директории.
        
    Returns:
        Путь к созданной директории.
    """
    # Создаем базовую директорию, если она не существует
    os.makedirs(base_dir, exist_ok=True)
    
    # Генерируем имя директории с временной меткой
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}"
    
    # Создаем директорию с временной меткой
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    
    return dir_path


class AverageMeter:
    """
    Класс для вычисления и хранения среднего значения и текущего значения.
    """
    
    def __init__(self, name: str):
        """
        Инициализация счетчика.
        
        Args:
            name: Название метрики.
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Сброс всех значений."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Обновление счетчика.
        
        Args:
            val: Новое значение.
            n: Вес нового значения.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 