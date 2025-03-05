#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорт наших модулей
from src.models import GaussianDiffusion, load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Генерация 3D моделей из текстовых описаний')
    
    parser.add_argument('--model_type', type=str, default='text300M', 
                        choices=['image300M', 'text300M', 'image1B', 'text1B'],
                        help='Тип модели для генерации')
    
    parser.add_argument('--text', type=str, default='стул в стиле минимализм', 
                        help='Текстовое описание для генерации 3D модели')
    
    parser.add_argument('--num_points', type=int, default=4096, 
                        help='Количество точек в генерируемом облаке точек')
    
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Размер батча для генерации')
    
    parser.add_argument('--num_samples', type=int, default=1, 
                        help='Количество сэмплов для генерации')
    
    parser.add_argument('--output', type=str, default='./outputs', 
                        help='Директория для сохранения результатов')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Устройство для генерации (cuda/cpu)')
    
    parser.add_argument('--seed', type=int, default=None, 
                        help='Seed для воспроизводимости результатов')
    
    return parser.parse_args()

def generate_point_cloud(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    text: str,
    device: torch.device,
    num_points: int = 4096,
) -> torch.Tensor:
    """
    Генерирует облако точек из текстового описания
    
    Args:
        model: модель для генерации
        diffusion: диффузионный процесс
        text: текстовое описание
        device: устройство для вычислений
        num_points: количество точек в облаке
        
    Returns:
        Тензор с облаком точек [1, num_points, 3]
    """
    # Формируем батч из одного текстового описания
    model_kwargs = {'texts': [text]}
    
    # Определяем форму выходного тензора
    shape = (1, num_points, 3)
    
    # Запускаем процесс сэмплирования
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model=model,
            shape=shape,
            model_kwargs=model_kwargs,
            device=device,
            progress=True
        )
    
    return sample

def save_point_cloud(
    points: torch.Tensor,
    output_path: Path,
    text: str,
    index: int = 0,
    visualize: bool = True
) -> None:
    """
    Сохраняет сгенерированное облако точек
    
    Args:
        points: тензор с облаком точек [1, num_points, 3]
        output_path: путь для сохранения
        text: текстовое описание
        index: индекс сэмпла
        visualize: генерировать ли визуализацию
    """
    # Создаем директорию, если её нет
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Извлекаем точки и переводим в numpy
    points_np = points[0].cpu().numpy()
    
    # Сохраняем точки в формате .npy
    points_file = output_path / f"sample_{index:03d}.npy"
    np.save(points_file, points_np)
    
    # Сохраняем текстовое описание
    text_file = output_path / f"sample_{index:03d}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Создаем визуализацию, если требуется
    if visualize:
        try:
            # Создаем 3D визуализацию
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1, alpha=0.5)
            ax.set_title(f"Sample {index}: {text}")
            ax.set_axis_off()
            
            # Сохраняем изображение
            plt_file = output_path / f"sample_{index:03d}.png"
            plt.savefig(plt_file, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # Более сложная визуализация с использованием Open3D, если доступно
            try:
                import open3d as o3d
                
                # Создаем объект облака точек
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_np)
                
                # Оцениваем нормали
                pcd.estimate_normals()
                
                # Сохраняем в формате PLY
                ply_file = output_path / f"sample_{index:03d}.ply"
                o3d.io.write_point_cloud(str(ply_file), pcd)
                
                logger.info(f"Сохранена 3D модель в {ply_file}")
            except ImportError:
                logger.warning("Библиотека Open3D не установлена, сохранение в формате PLY недоступно")
            
        except Exception as e:
            logger.error(f"Ошибка при создании визуализации: {e}")
    
    logger.info(f"Облако точек сохранено в {points_file}")

def main():
    args = parse_args()
    
    # Устанавливаем seed для воспроизводимости
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Определяем устройство
    device = torch.device(args.device)
    logger.info(f"Используется устройство: {device}")
    
    # Загружаем модель
    try:
        model = load_model(args.model_type, device=device)
        logger.info(f"Модель {args.model_type} загружена")
        
        # Создаем диффузионный процесс
        diffusion = GaussianDiffusion.from_pretrained('diffusion')
        logger.info("Диффузионный процесс инициализирован")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return
    
    # Создаем директорию для выходных данных
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Генерируем указанное количество сэмплов
    for i in range(args.num_samples):
        logger.info(f"Генерация сэмпла {i+1}/{args.num_samples}")
        
        try:
            # Генерируем облако точек
            points = generate_point_cloud(
                model=model,
                diffusion=diffusion,
                text=args.text,
                device=device,
                num_points=args.num_points
            )
            
            # Сохраняем результат
            save_point_cloud(
                points=points,
                output_path=output_path,
                text=args.text,
                index=i,
                visualize=True
            )
            
            # Освобождаем память
            del points
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Ошибка при генерации сэмпла {i+1}: {e}")
    
    logger.info(f"Генерация завершена. Сгенерировано {args.num_samples} моделей.")

if __name__ == "__main__":
    main() 