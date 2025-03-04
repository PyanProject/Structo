import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import random
import sys

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.voxelization import mesh_to_voxel

class ModelNetDataset(Dataset):
    """
    Датасет для работы с ModelNet40.
    """
    
    def __init__(self, root_dir, split='train', transform=None, voxel_resolution=64, text_augmentation=False):
        """
        Initializes the ModelNet dataset.
        
        Args:
            root_dir (str): Root directory of the dataset.
            split (str): 'train' or 'test' split.
            transform (callable, optional): Transform to apply to the data.
            voxel_resolution (int): Разрешение воксельной сетки.
            text_augmentation (bool): Whether to use text augmentation.
        """
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.voxel_resolution = voxel_resolution
        self.text_augmentation = text_augmentation
        self.file_list = []
        
        # Get all categories
        categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Load file paths
        for category in categories:
            category_dir = os.path.join(root_dir, category, split)
            
            if not os.path.exists(category_dir):
                continue
                
            files = sorted(os.listdir(category_dir))
            off_files = [f for f in files if f.endswith('.off')]
            
            for off_file in off_files:
                self.file_list.append({
                    'path': os.path.join(category_dir, off_file),
                    'category': category
                })
                
        print(f"Loaded {len(self.file_list)} files for {split} set")
    
    def __len__(self):
        """
        Returns the size of the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Получает элемент датасета по индексу.
        
        Args:
            idx (int): Индекс элемента.
            
        Returns:
            dict: Словарь с данными образца.
        """
        file_info = self.file_list[idx]
        file_path = file_info['path']
        category = file_info['category']
        
        # Загрузка меша
        try:
            mesh = trimesh.load(file_path)
            
            # Нормализация размера и позиции
            mesh.vertices -= mesh.bounding_box.centroid
            max_dim = np.max(mesh.bounding_box.extents)
            mesh.vertices /= max_dim
            
            # Преобразование в воксельное представление
            voxels = mesh_to_voxel(mesh, resolution=self.voxel_resolution)
            
            # Преобразование в тензор
            voxel_tensor = torch.FloatTensor(voxels).unsqueeze(0)  # [1, D, H, W]
            
            # Генерация текстового описания
            text_prompt = self.get_description(category)
            
            # Если нужны преобразования, применяем их
            if self.transform is not None:
                voxel_tensor = self.transform(voxel_tensor)
            
            return {
                'voxels': voxel_tensor,
                'text': text_prompt,
                'category': category,
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            # Возвращаем первый элемент в случае ошибки
            return self.__getitem__(0) if idx != 0 else None
    
    def get_description(self, category):
        """
        Generates a description for the 3D object category.
        
        Args:
            category (str): The category name.
            
        Returns:
            str: A text description of the 3D object.
        """
        # Basic description without augmentation
        if not self.text_augmentation or self.split == 'test':
            return f"3D model of object: {category}"
        
        # Choose a random template
        template = random.choice([
            "3D object: {}",
            "3D model of an object called {}",
            "three-dimensional model of {}",
            "{} in 3D space",
            "computer 3D model of: {}",
            "visualization of a three-dimensional item: {}",
            "3D render of: {}",
            "volumetric model of {}"
        ])
        
        # Choose a random attribute
        attribute = random.choice([
            "simple", "complex", "detailed", "geometric", 
            "common", "modern", "minimalist", "classic",
            "standard", "basic", "functional", "practical"
        ])
        
        # Add attribute with 50% probability
        if random.random() > 0.5:
            return template.format(f"{attribute} {category}")
        else:
            return template.format(category) 