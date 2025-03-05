import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import math

class VolumeRenderer(nn.Module):
    """Рендерер для создания 2D изображений из 3D представления."""
    
    def __init__(
        self,
        image_size: int = 256,
        n_pts_per_ray: int = 128,
        min_depth: float = 2.0,
        max_depth: float = 6.0,
        stratified: bool = True
    ):
        """
        Args:
            image_size: Размер выходного изображения
            n_pts_per_ray: Количество точек на луч
            min_depth: Минимальная глубина
            max_depth: Максимальная глубина
            stratified: Использовать ли стратифицированный сэмплинг
        """
        super().__init__()
        
        self.image_size = image_size
        self.n_pts_per_ray = n_pts_per_ray
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.stratified = stratified
        
    def get_rays(
        self,
        batch_size: int,
        device: torch.device,
        camera_to_world: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Генерация лучей для рендеринга.
        
        Args:
            batch_size: Размер батча
            device: Устройство для вычислений
            camera_to_world: Матрица трансформации камеры (опционально)
            
        Returns:
            ray_origins: Начальные точки лучей
            ray_directions: Направления лучей
        """
        # Создаем сетку пикселей
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, self.image_size),
            torch.linspace(-1, 1, self.image_size)
        )
        
        # Получаем направления лучей
        directions = torch.stack([j, -i, torch.ones_like(i)], dim=-1)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # Расширяем до батча
        directions = directions.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Применяем трансформацию камеры
        if camera_to_world is not None:
            directions = torch.einsum('bji,bhwj->bhwi', camera_to_world[:, :3, :3], directions)
            
        # Получаем начальные точки лучей
        origins = torch.zeros_like(directions)
        if camera_to_world is not None:
            origins += camera_to_world[:, None, None, :3, 3]
            
        return origins.to(device), directions.to(device)
        
    def sample_points(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Сэмплирование точек вдоль лучей.
        
        Args:
            ray_origins: Начальные точки лучей
            ray_directions: Направления лучей
            
        Returns:
            points: Сэмплированные точки
            z_vals: Значения глубины
        """
        # Получаем размеры
        batch_size, height, width, _ = ray_origins.shape
        device = ray_origins.device
        
        # Создаем значения глубины
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray)
        z_vals = z_vals.expand(batch_size, height, width, self.n_pts_per_ray)
        
        # Добавляем случайность при стратифицированном сэмплировании
        if self.stratified and self.training:
            z_vals = z_vals + torch.rand_like(z_vals) * (self.max_depth - self.min_depth) / self.n_pts_per_ray
            
        # Получаем точки
        points = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
        
        return points.to(device), z_vals.to(device)
        
    def render_rays(
        self,
        colors: torch.Tensor,
        densities: torch.Tensor,
        z_vals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Рендеринг лучей с помощью объемного рендеринга.
        
        Args:
            colors: Цвета точек
            densities: Плотности точек
            z_vals: Значения глубины
            
        Returns:
            rgb: Цвета пикселей
            depth: Карта глубины
        """
        # Получаем дельты между соседними точками
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], dim=-1)
        
        # Вычисляем альфа-композитинг
        alpha = 1.0 - torch.exp(-densities * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        # Вычисляем цвета и глубину
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        depth = torch.sum(weights * z_vals, dim=-1)
        
        return rgb, depth
        
    def forward(
        self,
        nerf_model: nn.Module,
        latent: torch.Tensor,
        camera_to_world: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через рендерер.
        
        Args:
            nerf_model: Модель NeRF
            latent: Латентный вектор
            camera_to_world: Матрица трансформации камеры
            
        Returns:
            rgb: Отрендеренное изображение
            depth: Карта глубины
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        # Получаем лучи
        ray_origins, ray_directions = self.get_rays(batch_size, device, camera_to_world)
        
        # Сэмплируем точки
        points, z_vals = self.sample_points(ray_origins, ray_directions)
        
        # Получаем цвета и плотности
        points_flat = points.view(batch_size, -1, 3)
        colors, densities = nerf_model(points_flat, latent)
        
        # Преобразуем обратно
        colors = colors.view(batch_size, self.image_size, self.image_size, self.n_pts_per_ray, 3)
        densities = densities.view(batch_size, self.image_size, self.image_size, self.n_pts_per_ray, 1)
        
        # Рендерим
        rgb, depth = self.render_rays(colors, densities, z_vals)
        
        return rgb, depth
        
    def create_camera_to_world(
        self,
        theta: float,
        phi: float,
        radius: float
    ) -> torch.Tensor:
        """
        Создание матрицы трансформации камеры.
        
        Args:
            theta: Угол поворота вокруг оси Y
            phi: Угол поворота вокруг оси X
            radius: Расстояние до центра
            
        Returns:
            Матрица трансформации 4x4
        """
        # Вычисляем позицию камеры
        x = radius * torch.sin(theta) * torch.cos(phi)
        y = radius * torch.sin(phi)
        z = radius * torch.cos(theta) * torch.cos(phi)
        
        # Создаем матрицу поворота
        camera_position = torch.tensor([x, y, z])
        look_at = torch.tensor([0.0, 0.0, 0.0])
        up = torch.tensor([0.0, 1.0, 0.0])
        
        z_axis = F.normalize(camera_position - look_at, dim=0)
        x_axis = F.normalize(torch.cross(up, z_axis), dim=0)
        y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0)
        
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)
        t = camera_position
        
        # Создаем матрицу трансформации
        camera_to_world = torch.eye(4)
        camera_to_world[:3, :3] = R
        camera_to_world[:3, 3] = t
        
        return camera_to_world
        
    def create_pan_cameras(
        self,
        n_views: int = 40,
        elevation: float = 0.0,
        radius: float = 4.0
    ) -> torch.Tensor:
        """
        Создание набора камер для кругового обзора.
        
        Args:
            n_views: Количество ракурсов
            elevation: Угол подъема камеры
            radius: Расстояние до центра
            
        Returns:
            Батч матриц трансформации
        """
        cameras = []
        for i in range(n_views):
            theta = 2 * math.pi * i / n_views
            camera_to_world = self.create_camera_to_world(theta, elevation, radius)
            cameras.append(camera_to_world)
            
        return torch.stack(cameras)
        
    def render_views(
        self,
        nerf_model: nn.Module,
        latent: torch.Tensor,
        n_views: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Рендеринг нескольких ракурсов модели.
        
        Args:
            nerf_model: Модель NeRF
            latent: Латентный вектор
            n_views: Количество ракурсов
            
        Returns:
            images: Отрендеренные изображения
            depths: Карты глубины
        """
        cameras = self.create_pan_cameras(n_views)
        cameras = cameras.to(latent.device)
        
        images = []
        depths = []
        
        for camera_to_world in cameras:
            camera_to_world = camera_to_world.unsqueeze(0).expand(latent.shape[0], -1, -1)
            rgb, depth = self.forward(nerf_model, latent, camera_to_world)
            images.append(rgb)
            depths.append(depth)
            
        return torch.stack(images), torch.stack(depths)
        
    def save_image(
        self,
        image: torch.Tensor,
        filename: str,
        normalize: bool = True
    ):
        """
        Сохранение изображения в файл.
        
        Args:
            image: Тензор изображения
            filename: Путь для сохранения
            normalize: Нормализовать ли значения
        """
        if normalize:
            image = (image - image.min()) / (image.max() - image.min())
            
        image = (image * 255).clamp(0, 255).byte()
        image = image.detach().cpu().numpy()
        
        if image.shape[-1] == 1:
            image = image.squeeze(-1)
            
        Image.fromarray(image).save(filename) 