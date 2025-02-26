import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import logging

logger = logging.getLogger(__name__)

def visualize_voxel_grid(voxel_grid, threshold=0.5, save_path=None, show=False):
    """
    Визуализирует воксельную сетку в виде 2D-срезов.
    
    Args:
        voxel_grid (numpy.ndarray or torch.Tensor): Воксельная сетка.
        threshold (float): Порог бинаризации для визуализации.
        save_path (str, optional): Путь для сохранения визуализации.
        show (bool): Отображать визуализацию интерактивно.
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры (если show=False).
    """
    # Преобразование в numpy, если нужно
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()
    
    # Удаление лишних размерностей
    if voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze()
    
    # Получение размерности сетки
    dim = voxel_grid.shape[0]
    
    # Выбор срезов для визуализации
    slice_indices = [dim // 4, dim // 2, 3 * dim // 4]
    
    # Создание фигуры
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Срезы воксельной сетки, порог: {threshold}')
    
    # Визуализация срезов
    for i, idx in enumerate(slice_indices):
        # Бинаризация для визуализации
        slice_data = voxel_grid[idx] > threshold
        
        im = axes[i].imshow(slice_data, cmap='viridis')
        axes[i].set_title(f'Срез {idx}/{dim}')
        axes[i].axis('off')
        
    # Сохранение, если указан путь
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Визуализация сохранена в {save_path}")
    
    # Отображение, если требуется
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def create_3d_visualization(voxel_grid, threshold=0.5, save_path=None, show=False):
    """
    Создает 3D-визуализацию воксельной сетки.
    
    Args:
        voxel_grid (numpy.ndarray or torch.Tensor): Воксельная сетка.
        threshold (float): Порог бинаризации для визуализации.
        save_path (str, optional): Путь для сохранения визуализации.
        show (bool): Отображать визуализацию интерактивно.
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры (если show=False).
    """
    # Преобразование в numpy, если нужно
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()
    
    # Удаление лишних размерностей
    if voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze()
    
    # Бинаризация воксельной сетки
    binary_voxels = voxel_grid > threshold
    
    # Проверка наличия активных вокселей
    if np.sum(binary_voxels) == 0:
        logger.warning("Воксельная сетка пуста после бинаризации!")
        # Создаем пустую фигуру
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Воксельная сетка пуста!")
        
        # Сохранение, если указан путь
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Визуализация сохранена в {save_path}")
        
        # Отображение, если требуется
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    # Получение координат активных вокселей
    voxels = np.where(binary_voxels)
    x, y, z = voxels[0], voxels[1], voxels[2]
    
    # Ограничение количества вокселей для визуализации
    max_voxels = 10000
    if len(x) > max_voxels:
        logger.info(f"Ограничение количества вокселей для визуализации: {max_voxels} из {len(x)}")
        indices = np.random.choice(len(x), max_voxels, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    
    # Создание цветов на основе значений вокселей
    if np.sum(binary_voxels) > 0:
        values = voxel_grid[x, y, z]
        normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        colors = plt.cm.viridis(normalized_values)
    else:
        colors = 'blue'
    
    # Создание 3D-визуализации
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Отображение вокселей как точек
    ax.scatter(x, y, z, c=colors, marker='o', alpha=0.7, s=50)
    
    # Настройка вида
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D-визуализация воксельной сетки, порог: {threshold}')
    
    # Масштабирование осей
    dim = voxel_grid.shape[0]
    ax.set_xlim(0, dim)
    ax.set_ylim(0, dim)
    ax.set_zlim(0, dim)
    
    # Сохранение, если указан путь
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"3D-визуализация сохранена в {save_path}")
    
    # Отображение, если требуется
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def create_multiview_visualization(voxel_grid, threshold=0.5, save_path=None, show=False):
    """
    Создает визуализацию воксельной сетки с нескольких ракурсов.
    
    Args:
        voxel_grid (numpy.ndarray or torch.Tensor): Воксельная сетка.
        threshold (float): Порог бинаризации для визуализации.
        save_path (str, optional): Путь для сохранения визуализации.
        show (bool): Отображать визуализацию интерактивно.
        
    Returns:
        matplotlib.figure.Figure: Объект фигуры (если show=False).
    """
    # Преобразование в numpy, если нужно
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()
    
    # Удаление лишних размерностей
    if voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze()
    
    # Бинаризация воксельной сетки
    binary_voxels = voxel_grid > threshold
    
    # Проверка наличия активных вокселей
    if np.sum(binary_voxels) == 0:
        logger.warning("Воксельная сетка пуста после бинаризации!")
        # Создаем пустую фигуру
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        ax.set_title("Воксельная сетка пуста!")
        ax.axis('off')
        
        # Сохранение, если указан путь
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Визуализация сохранена в {save_path}")
        
        # Отображение, если требуется
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    # Получение координат активных вокселей
    voxels = np.where(binary_voxels)
    x, y, z = voxels[0], voxels[1], voxels[2]
    
    # Ограничение количества вокселей для визуализации
    max_voxels = 5000
    if len(x) > max_voxels:
        logger.info(f"Ограничение количества вокселей для визуализации: {max_voxels} из {len(x)}")
        indices = np.random.choice(len(x), max_voxels, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    
    # Создание цветов на основе значений вокселей
    values = voxel_grid[x, y, z]
    normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    colors = plt.cm.viridis(normalized_values)
    
    # Создание многовидовой визуализации
    fig = plt.figure(figsize=(15, 5))
    
    # Ракурсы для визуализации (фронтальный, сверху, сбоку)
    views = [(30, 30), (90, 0), (0, 0)]
    titles = ['Изометрический вид', 'Вид сверху', 'Вид сбоку']
    
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Отображение вокселей как точек
        ax.scatter(x, y, z, c=colors, marker='o', alpha=0.7, s=30)
        
        # Настройка вида
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[i])
        
        # Отключение осей для лучшей визуализации
        ax.set_axis_off()
        
        # Масштабирование осей
        dim = voxel_grid.shape[0]
        ax.set_xlim(0, dim)
        ax.set_ylim(0, dim)
        ax.set_zlim(0, dim)
    
    plt.tight_layout()
    
    # Сохранение, если указан путь
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Многовидовая визуализация сохранена в {save_path}")
    
    # Отображение, если требуется
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig 