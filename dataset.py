import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import trimesh
import numpy as np

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.file_list = self._load_file_list()
    
    def _load_file_list(self):
        split_file = os.path.join(self.root_dir, f'{self.split}_files.txt')
        with open(split_file, 'r') as f:
            file_paths = [line.strip() for line in f.readlines()]
        return file_paths
    
    def _load_mesh(self, file_path):
        mesh = trimesh.load(file_path, process=True)
        points = mesh.sample(1024)  # Пример: выборка 1024 точек из модели
        return points
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        point_cloud = self._load_mesh(file_path)
        
        if self.transform:
            point_cloud = self.transform(point_cloud)
        
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        return point_cloud
    
    def generate_dataset(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

def get_dataloader(root_dir, batch_size=32, split='train', shuffle=True, num_workers=4):
    dataset = ModelNet40Dataset(root_dir=root_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
