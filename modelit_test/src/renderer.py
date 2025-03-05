#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
import mcubes
import trimesh
from PIL import Image

class VolumeRenderer:
    """
    Рендерер объемных моделей для визуализации и экспорта 3D-моделей.
    """
    
    def __init__(
        self,
        device: torch.device,
        resolution: int = 128,
        threshold: float = 0.5,
        camera_dist: float = 2.5,
        num_views: int = 8,
    ):
        """
        Инициализация рендерера.
        
        Args:
            device: Устройство для вычислений (CPU/GPU).
            resolution: Разрешение воксельной сетки.
            threshold: Порог для извлечения поверхности из воксельного представления.
            camera_dist: Расстояние камеры от объекта.
            num_views: Количество видов для рендеринга.
        """
        self.device = device
        self.resolution = resolution
        self.threshold = threshold
        self.camera_dist = camera_dist
        self.num_views = num_views
        
    def get_camera_positions(self) -> torch.Tensor:
        """
        Получение позиций камеры для рендеринга нескольких видов.
        
        Returns:
            Тензор позиций камеры формы [num_views, 3].
        """
        # Распределяем камеры равномерно по кругу в плоскости XZ
        theta = torch.linspace(0, 2 * np.pi, self.num_views + 1)[:-1]
        x = self.camera_dist * torch.cos(theta)
        z = self.camera_dist * torch.sin(theta)
        y = torch.zeros_like(x) + self.camera_dist * 0.5  # Немного приподнимаем камеру
        
        positions = torch.stack([x, y, z], dim=1).to(self.device)
        return positions
    
    def get_camera_look_at(
        self,
        camera_positions: torch.Tensor,
        target: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Вычисление направлений взгляда камеры.
        
        Args:
            camera_positions: Позиции камеры формы [num_views, 3].
            target: Целевая точка для всех камер. По умолчанию [0, 0, 0].
            
        Returns:
            Тензор направлений камеры формы [num_views, 3].
        """
        if target is None:
            target = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        # Для каждой позиции камеры вычисляем направление взгляда
        look_at = target.unsqueeze(0) - camera_positions
        
        # Нормализуем направления
        look_at = look_at / torch.norm(look_at, dim=1, keepdim=True)
        
        return look_at
    
    def voxel_grid_to_mesh(
        self,
        voxel_grid: torch.Tensor,
        smooth: bool = True
    ) -> trimesh.Trimesh:
        """
        Преобразование воксельной сетки в полигональную сетку с помощью marching cubes.
        
        Args:
            voxel_grid: Воксельная сетка формы [depth, height, width].
            smooth: Флаг сглаживания сетки.
            
        Returns:
            Полигональная сетка (trimesh.Trimesh).
        """
        # Переводим воксели в numpy для работы с mcubes
        voxels_np = voxel_grid.detach().cpu().numpy()
        
        # Извлекаем поверхность с помощью marching cubes
        vertices, triangles = mcubes.marching_cubes(voxels_np, self.threshold)
        
        # Центрируем координаты
        vertices = vertices / self.resolution - 0.5
        
        # Создаем сетку
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        # Сглаживаем сетку, если нужно
        if smooth:
            mesh = mesh.smoothed()
        
        return mesh
    
    def render_views(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 256,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> List[Image.Image]:
        """
        Рендеринг нескольких видов сетки.
        
        Args:
            mesh: Полигональная сетка для рендеринга.
            resolution: Разрешение выходных изображений.
            background_color: Цвет фона.
            
        Returns:
            Список изображений с разных ракурсов.
        """
        # Получаем позиции камеры
        camera_positions = self.get_camera_positions().cpu().numpy()
        
        # Рендерим виды
        images = []
        for i in range(self.num_views):
            # Расположение камеры
            camera_pos = camera_positions[i]
            
            # Направление взгляда (к началу координат)
            camera_target = np.array([0.0, 0.0, 0.0])
            camera_direction = camera_target - camera_pos
            camera_direction = camera_direction / np.linalg.norm(camera_direction)
            
            # Направление вверх для камеры
            camera_up = np.array([0.0, 1.0, 0.0])
            
            # Создаем сцену для рендеринга
            scene = mesh.scene()
            
            # Устанавливаем камеру
            scene.camera.fov = 60  # Угол обзора в градусах
            scene.camera.resolution = [resolution, resolution]
            
            # Устанавливаем матрицу преобразования камеры
            rotation = trimesh.transformations.rotation_matrix(
                np.radians(180),  # Поворот на 180 градусов вокруг Y оси
                [0, 1, 0],
                point=camera_target
            )
            
            # Устанавливаем позицию и ориентацию камеры
            scene.camera_transform = trimesh.geometry.align_vectors(
                np.array([0, 0, 1]),  # Направление взгляда по умолчанию
                camera_direction,
                direction_B=camera_up,
                point=camera_pos
            ) @ rotation
            
            # Рендерим сцену
            rendered_image = scene.save_image(
                resolution=resolution,
                background=background_color
            )
            
            # Преобразуем массив байтов в изображение PIL
            image = Image.open(trimesh.util.BytesIO(rendered_image))
            images.append(image)
        
        return images
    
    def extract_mesh_from_voxels(
        self,
        voxels: torch.Tensor,
        smooth: bool = True
    ) -> Tuple[trimesh.Trimesh, List[Image.Image]]:
        """
        Извлечение полигональной сетки из воксельного представления и рендеринг видов.
        
        Args:
            voxels: Воксельная сетка формы [1, 1, depth, height, width].
            smooth: Флаг сглаживания сетки.
            
        Returns:
            Кортеж (mesh, views), где mesh - полигональная сетка,
            а views - список изображений с разных ракурсов.
        """
        # Убираем ненужные размерности
        voxels = voxels.squeeze(0).squeeze(0)
        
        # Извлекаем меш
        mesh = self.voxel_grid_to_mesh(voxels, smooth)
        
        # Рендерим виды
        views = self.render_views(mesh)
        
        return mesh, views
    
    def save_mesh(
        self,
        mesh: trimesh.Trimesh,
        file_path: str,
        file_type: str = 'obj'
    ) -> None:
        """
        Сохранение полигональной сетки в файл.
        
        Args:
            mesh: Полигональная сетка для сохранения.
            file_path: Путь для сохранения файла.
            file_type: Тип файла (obj, stl, etc.).
        """
        mesh.export(file_path, file_type=file_type)
    
    def create_gif(
        self,
        images: List[Image.Image],
        file_path: str,
        duration: int = 100
    ) -> None:
        """
        Создание GIF-анимации из серии изображений.
        
        Args:
            images: Список изображений для анимации.
            file_path: Путь для сохранения GIF.
            duration: Длительность каждого кадра в миллисекундах.
        """
        # Сохраняем первое изображение
        images[0].save(
            file_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False
        ) 