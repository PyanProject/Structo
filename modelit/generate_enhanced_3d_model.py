#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import torch
import numpy as np
import argparse
import logging
import trimesh
from skimage import measure
from scipy import ndimage
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Добавляем путь к пакету src в sys.path
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.append(current_path)

from src.models.text_to_3d_model import TextTo3DModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Загружает конфигурацию из YAML файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Преобразование словаря в объект с атрибутами
    config = SimpleNamespace(**config_dict)
    
    # Преобразование вложенных словарей в объекты
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(config, key, SimpleNamespace(**value))
            
            # Обработка вложенных словарей второго уровня
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    setattr(getattr(config, key), subkey, SimpleNamespace(**subvalue))
    
    return config

def save_mesh(mesh, path):
    """Сохраняет меш в файл."""
    try:
        mesh.export(path)
        logger.info(f"Меш успешно сохранен в {path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении меша: {str(e)}")
        return False

def visualize_voxel_stats(voxel_grid, threshold=0.5, save_path=None):
    """Визуализация статистики воксельной сетки и построение гистограммы."""
    logger.info(f"Статистика воксельной сетки:")
    logger.info(f"Форма: {voxel_grid.shape}")
    logger.info(f"Минимум: {voxel_grid.min():.6f}, Максимум: {voxel_grid.max():.6f}")
    logger.info(f"Среднее: {voxel_grid.mean():.6f}, Медиана: {np.median(voxel_grid):.6f}")
    
    # Подсчет активных вокселей
    binary_grid = voxel_grid > threshold
    active_voxels = np.sum(binary_grid)
    total_voxels = binary_grid.size
    logger.info(f"Активных вокселей при пороге {threshold}: {active_voxels}/{total_voxels} ({active_voxels/total_voxels*100:.4f}%)")
    
    # Построение гистограммы
    plt.figure(figsize=(12, 6))
    
    # Общая гистограмма
    plt.subplot(1, 2, 1)
    plt.hist(voxel_grid.flatten(), bins=50, alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Порог ({threshold})')
    plt.xlabel('Значение вокселя')
    plt.ylabel('Частота')
    plt.title('Гистограмма значений вокселей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Гистограмма для активных вокселей
    plt.subplot(1, 2, 2)
    active_values = voxel_grid[binary_grid]
    if len(active_values) > 0:
        plt.hist(active_values, bins=20, alpha=0.7, color='green')
        plt.xlabel('Значение активного вокселя')
        plt.ylabel('Частота')
        plt.title(f'Гистограмма активных вокселей (>{threshold})')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Нет активных вокселей', ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Сохранение гистограммы, если указан путь
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Гистограмма сохранена в {save_path}")
    
    return active_voxels, total_voxels

def visualize_voxel_slices(voxel_grid, save_path=None, num_slices=5):
    """Визуализация срезов воксельной сетки."""
    # Определение индексов срезов
    depth = voxel_grid.shape[0]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    plt.figure(figsize=(15, 3*num_slices))
    
    for i, idx in enumerate(slice_indices):
        plt.subplot(num_slices, 1, i+1)
        plt.imshow(voxel_grid[idx], cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f'Срез {idx}/{depth-1}')
        plt.grid(False)
    
    plt.tight_layout()
    
    # Сохранение срезов, если указан путь
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Срезы сохранены в {save_path}")

def voxel_to_mesh(voxels, threshold=0.5, smooth=True, method='marching_cubes'):
    """
    Преобразует воксельную сетку в меш.
    
    Args:
        voxels (numpy.ndarray): Воксельная сетка.
        threshold (float): Порог бинаризации.
        smooth (bool): Применять ли сглаживание.
        method (str): Метод генерации меша ('marching_cubes' или 'cubes').
        
    Returns:
        trimesh.Trimesh: Меш.
    """
    try:
        # Применение сглаживания, если требуется
        if smooth:
            smoothed_voxels = ndimage.gaussian_filter(voxels, sigma=0.5)
        else:
            smoothed_voxels = voxels
        
        # Бинаризация воксельной сетки
        binary_voxels = smoothed_voxels > threshold
        
        active_voxels = np.sum(binary_voxels)
        total_voxels = binary_voxels.size
        logger.info(f"Активных вокселей после бинаризации: {active_voxels}/{total_voxels} ({active_voxels/total_voxels*100:.4f}%)")
        
        # Если нет активных вокселей, возвращаем простой куб
        if active_voxels == 0:
            logger.warning("Воксельная сетка пуста после бинаризации. Возвращается базовый куб.")
            return trimesh.creation.box(extents=[1, 1, 1])
        
        if method == 'marching_cubes':
            # Создание меша с использованием marching cubes
            vertices, faces, normals, _ = measure.marching_cubes(smoothed_voxels, level=threshold)
            
            # Создание меша из trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            
            logger.info(f"Создан меш с {len(mesh.vertices)} вершинами и {len(mesh.faces)} гранями.")
            
            # Проверка на наличие ошибок в меше
            if not mesh.is_watertight:
                logger.warning("Меш не является водонепроницаемым. Возможны проблемы с геометрией.")
            
            if len(mesh.faces) < 4:
                logger.warning("Меш имеет очень мало граней. Может потребоваться другой метод генерации.")
            
            return mesh
            
        elif method == 'cubes':
            # Альтернативный метод: создаем кубы для каждого активного вокселя
            logger.info("Использование метода создания кубов для каждого активного вокселя")
            
            # Находим координаты активных вокселей
            voxel_coords = np.where(binary_voxels)
            logger.info(f"Найдено {len(voxel_coords[0])} активных вокселей.")
            
            # Ограничиваем количество кубов для производительности
            max_cubes = min(1000, len(voxel_coords[0]))
            
            meshes = []
            for i in range(max_cubes):
                box = trimesh.creation.box(extents=[1, 1, 1])
                box.apply_translation([voxel_coords[0][i], voxel_coords[1][i], voxel_coords[2][i]])
                meshes.append(box)
            
            # Объединяем все кубы в один меш
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
                logger.info(f"Создан меш из {len(meshes)} кубов: {len(mesh.vertices)} вершин и {len(mesh.faces)} граней.")
                return mesh
            else:
                return trimesh.creation.box(extents=[1, 1, 1])
        else:
            raise ValueError(f"Неизвестный метод создания меша: {method}")
                
    except Exception as e:
        logger.error(f"Ошибка при создании меша: {str(e)}")
        logger.info("Возвращается базовый куб.")
        return trimesh.creation.box(extents=[1, 1, 1])

def generate_model(model, text_prompt, output_dir, base_filename, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5), device="cpu"):
    """
    Генерирует 3D модель по текстовому описанию с различными порогами.
    
    Args:
        model: Модель для генерации.
        text_prompt (str): Текстовое описание.
        output_dir (str): Директория для сохранения результатов.
        base_filename (str): Базовое имя файла без расширения.
        thresholds (list): Список порогов бинаризации.
        device (str): Устройство для вычислений.
    """
    logger.info(f"Генерация 3D модели для промпта: '{text_prompt}'")
    
    # Создание директории для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Создание директории для визуализаций
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    try:
        # Переводим модель в режим оценки
        model.eval()
        
        # Генерация воксельной сетки
        with torch.no_grad():
            voxel_grid = model(text_prompt)
            logger.info(f"Получен тензор формы {voxel_grid.shape}")
            
            # Преобразование в numpy массив и применение сигмоиды
            raw_voxel_array = voxel_grid[0, 0].detach().cpu().numpy()
            voxel_array = torch.sigmoid(voxel_grid[0, 0]).detach().cpu().numpy()
            
            # Визуализация статистики воксельной сетки
            hist_path = os.path.join(visualizations_dir, f"{base_filename}_histogram.png")
            visualize_voxel_stats(voxel_array, threshold=0.5, save_path=hist_path)
            
            # Визуализация срезов воксельной сетки
            slices_path = os.path.join(visualizations_dir, f"{base_filename}_slices.png")
            visualize_voxel_slices(voxel_array, save_path=slices_path)
            
            # Сохранение меша с разными порогами и методами
            for threshold in thresholds:
                # Marching Cubes метод
                try:
                    mesh_mc = voxel_to_mesh(voxel_array, threshold=threshold, method='marching_cubes')
                    mc_output_path = os.path.join(output_dir, f"{base_filename}_thresh_{threshold:.2f}_mc.obj")
                    save_mesh(mesh_mc, mc_output_path)
                except Exception as e:
                    logger.error(f"Ошибка при генерации меша с методом marching_cubes и порогом {threshold}: {str(e)}")
                
                # Метод кубов
                try:
                    mesh_cubes = voxel_to_mesh(voxel_array, threshold=threshold, method='cubes')
                    cubes_output_path = os.path.join(output_dir, f"{base_filename}_thresh_{threshold:.2f}_cubes.obj")
                    save_mesh(mesh_cubes, cubes_output_path)
                except Exception as e:
                    logger.error(f"Ошибка при генерации меша с методом cubes и порогом {threshold}: {str(e)}")
            
            # Сохранение сетки voxel_array в numpy формате для дальнейшего анализа
            np_output_path = os.path.join(output_dir, f"{base_filename}_voxel_grid.npy")
            np.save(np_output_path, voxel_array)
            logger.info(f"Воксельная сетка сохранена в {np_output_path}")
            
            return voxel_array
    
    except Exception as e:
        logger.error(f"Ошибка при генерации модели: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Основная функция для генерации 3D моделей.
    """
    parser = argparse.ArgumentParser(description="Генерация 3D моделей из текстовых описаний")
    parser.add_argument("--config", type=str, default="modelit/configs/enhanced_architecture.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпоинту модели")
    parser.add_argument("--prompt", type=str, default="3D model of an airplane", help="Текстовое описание для генерации 3D модели")
    parser.add_argument("--output_dir", type=str, default="models/enhanced", help="Директория для сохранения результатов")
    parser.add_argument("--filename", type=str, default=None, help="Базовое имя файла (без расширения)")
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5", help="Пороги бинаризации, разделенные запятыми")
    parser.add_argument("--device", type=str, default=None, help="Устройство для вычислений (cuda/cpu)")
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    logger.info(f"Конфигурация загружена из {args.config}")
    
    # Определение устройства
    device_str = args.device if args.device else getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    logger.info(f"Используется устройство: {device}")
    
    # Инициализация модели
    model = TextTo3DModel(config)
    logger.info("Модель инициализирована")
    
    # Загрузка весов
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Веса модели загружены из {args.checkpoint}")
    
    # Перенос модели на устройство
    model = model.to(device)
    
    # Парсинг порогов
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    # Создание базового имени файла, если не указано
    if args.filename is None:
        # Преобразование промпта в имя файла
        filename = args.prompt.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        # Ограничение длины имени файла
        if len(filename) > 50:
            filename = filename[:50]
    else:
        filename = args.filename
    
    # Генерация модели
    _ = generate_model(model, args.prompt, args.output_dir, filename, thresholds, device)
    
    logger.info("Генерация завершена")

if __name__ == "__main__":
    main() 