'''
этот файл подгружает датасет для дальнейшей работы йоу

'''


import os
import torch
from torch.utils.data import Dataset
import numpy as np

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.file_list = self._get_file_list()
    
    def _get_file_list(self):
        file_list = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name, self.split)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.endswith(".off"):
                        file_list.append(os.path.join(class_path, file_name))
        return file_list
    
    def _load_off_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines[0].strip() != 'OFF':
                raise ValueError('Not a valid OFF file')
            n_verts, n_faces, _ = map(int, lines[1].strip().split())
            verts = [list(map(float, line.strip().split())) for line in lines[2:2 + n_verts]]
            faces = [list(map(int, line.strip().split()[1:])) for line in lines[2 + n_verts:2 + n_verts + n_faces]]
            return np.array(verts), np.array(faces)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            vertices, faces = self._load_off_file(file_path)
            if self.transform:
                vertices = self.transform(vertices)
            return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)
        except Exception as e:
            print(f"Файл {file_path} повреждён или не может быть загружен: {e}")
            return None  # Возвращаем None вместо рекурсии

# dataset.py
def collate_fn(batch):
    valid_items = [item for item in batch if item is not None]
    
    # Если все элементы в батче повреждены, пропускаем его
    if not valid_items:
        return None, None
    
    # Фиксированный размер вершин (1024 точки)
    fixed_num_points = 1024
    padded_batch = []
    faces_batch = []
    
    for item in valid_items:
        vertices, faces = item
        current_num_points = vertices.size(0)
        
        # Обрезка или дополнение нулями до 1024 точек
        if current_num_points >= fixed_num_points:
            padded_item = vertices[:fixed_num_points]
        else:
            padding = torch.zeros((fixed_num_points - current_num_points, 3))
            padded_item = torch.cat((vertices, padding), dim=0)
        
        padded_batch.append(padded_item)
        faces_batch.append(faces)
    
    return torch.stack(padded_batch), faces_batch