import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
import trimesh

class Decoder3D(nn.Module):
    """Декодер для преобразования латентного представления в 3D модель."""
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dims: List[int] = [512, 256, 128, 64],
        voxel_size: int = 32,
        feature_size: int = 32,
    ):
        """
        Args:
            latent_dim: Размерность латентного пространства
            hidden_dims: Размерности скрытых слоев
            voxel_size: Размер воксельной сетки
            feature_size: Размер признакового пространства
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.voxel_size = voxel_size
        self.feature_size = feature_size
        
        # Создаем слои декодера
        layers = []
        in_features = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_features = hidden_dim
            
        self.shared = nn.Sequential(*layers)
        
        # Ветвь для предсказания занятости вокселей
        self.occupancy_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], voxel_size**3),
            nn.Sigmoid()
        )
        
        # Ветвь для предсказания признаков
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], voxel_size**3 * feature_size),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через декодер.
        
        Args:
            z: Латентный вектор формы (batch_size, latent_dim)
            
        Returns:
            occupancy: Вероятности занятости вокселей (batch_size, voxel_size, voxel_size, voxel_size)
            features: Признаки вокселей (batch_size, voxel_size, voxel_size, voxel_size, feature_size)
        """
        batch_size = z.shape[0]
        
        # Общие признаки
        shared_features = self.shared(z)
        
        # Предсказание занятости
        occupancy = self.occupancy_head(shared_features)
        occupancy = occupancy.view(batch_size, self.voxel_size, self.voxel_size, self.voxel_size)
        
        # Предсказание признаков
        features = self.feature_head(shared_features)
        features = features.view(batch_size, self.voxel_size, self.voxel_size, self.voxel_size, self.feature_size)
        
        return occupancy, features
        
    def extract_mesh(
        self,
        occupancy: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        smooth: bool = True
    ) -> trimesh.Trimesh:
        """
        Извлечение меша из воксельного представления.
        
        Args:
            occupancy: Вероятности занятости вокселей
            features: Признаки вокселей (опционально)
            threshold: Порог для определения занятых вокселей
            smooth: Применять ли сглаживание к мешу
            
        Returns:
            Полигональный меш
        """
        # Переводим в numpy
        occupancy = occupancy.detach().cpu().numpy()
        if features is not None:
            features = features.detach().cpu().numpy()
            
        # Создаем воксельную сетку
        voxels = occupancy > threshold
        
        # Извлекаем меш с помощью marching cubes
        from skimage import measure
        vertices, faces, normals, values = measure.marching_cubes(
            occupancy,
            level=threshold,
            spacing=(1.0, 1.0, 1.0)
        )
        
        # Создаем меш
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        
        # Сглаживание
        if smooth:
            mesh = mesh.smoothed()
            
        return mesh
        
    def save_mesh(self, mesh: trimesh.Trimesh, filename: str):
        """
        Сохранение меша в файл.
        
        Args:
            mesh: Полигональный меш
            filename: Путь для сохранения
        """
        mesh.export(filename)
        
class NeRFDecoder(nn.Module):
    """Декодер на основе NeRF для создания 3D представления."""
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_frequencies: int = 10,
        skip_connections: List[int] = [4],
    ):
        """
        Args:
            latent_dim: Размерность латентного пространства
            hidden_dim: Размерность скрытых слоев
            num_layers: Количество слоев MLP
            num_frequencies: Количество частот для позиционного кодирования
            skip_connections: Слои со skip-соединениями
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.skip_connections = skip_connections
        
        # Размерность позиционного кодирования
        self.pos_dim = 3 * (2 * num_frequencies + 1)
        
        # Создаем слои
        self.layers = nn.ModuleList()
        
        # Первый слой
        self.layers.append(nn.Linear(self.pos_dim + latent_dim, hidden_dim))
        
        # Промежуточные слои
        for i in range(num_layers - 1):
            if i in skip_connections:
                self.layers.append(nn.Linear(hidden_dim + self.pos_dim + latent_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                
        # Выходные слои
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
        
    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Позиционное кодирование координат.
        
        Args:
            x: Тензор координат (batch_size, num_points, 3)
            
        Returns:
            Позиционно закодированные координаты
        """
        encodings = [x]
        
        for i in range(self.num_frequencies):
            freq = 2.0 ** i
            for func in [torch.sin, torch.cos]:
                encodings.append(func(freq * x))
                
        return torch.cat(encodings, dim=-1)
        
    def forward(
        self,
        points: torch.Tensor,
        latent: torch.Tensor,
        return_density: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Прямой проход через декодер.
        
        Args:
            points: Координаты точек (batch_size, num_points, 3)
            latent: Латентный вектор (batch_size, latent_dim)
            return_density: Возвращать ли плотность
            
        Returns:
            colors: Цвета точек (batch_size, num_points, 3)
            density: Плотность точек (batch_size, num_points, 1) если return_density=True
        """
        batch_size, num_points, _ = points.shape
        
        # Позиционное кодирование координат
        x = self.positional_encoding(points)
        
        # Расширяем латентный вектор
        latent = latent.unsqueeze(1).expand(-1, num_points, -1)
        
        # Конкатенируем координаты и латентный вектор
        x = torch.cat([x, latent], dim=-1)
        
        # Проходим через слои
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, self.positional_encoding(points), latent], dim=-1)
            x = F.relu(layer(x))
            
        # Получаем цвета и плотность
        colors = self.color_head(x)
        
        if return_density:
            density = self.density_head(x)
            return colors, density
        
        return colors, None 