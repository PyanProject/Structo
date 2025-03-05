import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import trimesh
import numpy as np
from typing import Optional, Dict, List, Tuple
from PIL import Image
import logging

class MeshDataset(Dataset):
    """Датасет для работы с 3D мешами и их текстовыми описаниями."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        voxel_size: int = 32,
        num_views: int = 8,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data_dir: Путь к директории с данными
            split: Разбиение датасета ('train', 'val', 'test')
            voxel_size: Размер воксельной сетки
            num_views: Количество ракурсов для рендеринга
            transform: Функция преобразования данных
        """
        self.data_dir = data_dir
        self.split = split
        self.voxel_size = voxel_size
        self.num_views = num_views
        self.transform = transform
        
        # Настройка логгера
        self.logger = logging.getLogger(__name__)
        
        # Загрузка метаданных
        self.metadata = self.load_metadata()
        
        # Создание индекса
        self.samples = self.create_index()
        
    def load_metadata(self) -> Dict:
        """
        Загрузка метаданных датасета.
        
        Returns:
            Словарь с метаданными
        """
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f'Файл метаданных не найден: {metadata_path}')
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        return metadata
        
    def create_index(self) -> List[Dict]:
        """
        Создание индекса датасета.
        
        Returns:
            Список словарей с информацией о сэмплах
        """
        split_file = os.path.join(self.data_dir, f'{self.split}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f'Файл разбиения не найден: {split_file}')
            
        with open(split_file, 'r') as f:
            sample_ids = [line.strip() for line in f]
            
        samples = []
        for sample_id in sample_ids:
            if sample_id in self.metadata:
                sample_data = {
                    'id': sample_id,
                    'mesh_path': os.path.join(self.data_dir, 'meshes', f'{sample_id}.obj'),
                    'text': self.metadata[sample_id]['text'],
                    'category': self.metadata[sample_id].get('category', None)
                }
                samples.append(sample_data)
            else:
                self.logger.warning(f'Метаданные не найдены для образца: {sample_id}')
                
        return samples
        
    def load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """
        Загрузка меша из файла.
        
        Args:
            mesh_path: Путь к файлу меша
            
        Returns:
            Загруженный меш
        """
        try:
            mesh = trimesh.load(mesh_path)
            return mesh
        except Exception as e:
            self.logger.error(f'Ошибка при загрузке меша {mesh_path}: {str(e)}')
            raise
            
    def voxelize_mesh(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """
        Преобразование меша в воксельное представление.
        
        Args:
            mesh: Полигональный меш
            
        Returns:
            Воксельная сетка
        """
        # Нормализация меша
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        mesh.vertices = mesh.vertices / np.abs(mesh.vertices).max()
        
        # Воксилизация
        voxels = mesh.voxelized(pitch=1.0/self.voxel_size)
        voxel_grid = voxels.matrix.astype(np.float32)
        
        return torch.from_numpy(voxel_grid)
        
    def render_views(
        self,
        mesh: trimesh.Trimesh,
        size: int = 224
    ) -> torch.Tensor:
        """
        Рендеринг нескольких ракурсов меша.
        
        Args:
            mesh: Полигональный меш
            size: Размер изображения
            
        Returns:
            Тензор с изображениями
        """
        # Создаем сцену
        scene = trimesh.Scene(mesh)
        
        # Получаем углы для камер
        angles = np.linspace(0, 2*np.pi, self.num_views)
        
        # Рендерим виды
        views = []
        for angle in angles:
            # Матрица поворота камеры
            camera_transform = trimesh.transformations.rotation_matrix(
                angle,
                [0, 1, 0],
                scene.centroid
            )
            
            # Рендерим
            img = scene.save_image(
                resolution=(size, size),
                camera_transform=camera_transform
            )
            
            # Преобразуем в тензор
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            views.append(img)
            
        return torch.stack(views)
        
    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Получение элемента датасета.
        
        Args:
            idx: Индекс элемента
            
        Returns:
            Словарь с данными образца
        """
        sample = self.samples[idx]
        
        # Загружаем меш
        mesh = self.load_mesh(sample['mesh_path'])
        
        # Получаем воксельное представление
        voxels = self.voxelize_mesh(mesh)
        
        # Рендерим виды
        views = self.render_views(mesh)
        
        return {
            'id': sample['id'],
            'text': sample['text'],
            'category': sample['category'],
            'voxels': voxels,
            'views': views
        }
        
def create_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Создание загрузчика данных.
    
    Args:
        data_dir: Путь к директории с данными
        split: Разбиение датасета
        batch_size: Размер батча
        num_workers: Количество процессов для загрузки данных
        **kwargs: Дополнительные аргументы для MeshDataset
        
    Returns:
        Загрузчик данных
    """
    dataset = MeshDataset(data_dir, split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader 