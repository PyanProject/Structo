import os
import torch
from torch.utils.data import Dataset
import numpy as np

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.class_to_idx = {}
        self.file_list = self._get_file_list()
    
    def _get_file_list(self):
        file_list = []
        for class_name in os.listdir(self.root_dir):
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            class_path = os.path.join(self.root_dir, class_name, self.split)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.endswith(".off"):
                        file_list.append((os.path.join(class_path, file_name), class_name))
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
        file_path, class_name = self.file_list[idx]
        try:
            vertices, faces = self._load_off_file(file_path)
            if self.transform:
                vertices = self.transform(vertices)
            return (
                torch.tensor(vertices, dtype=torch.float32), 
                torch.tensor(faces, dtype=torch.int64),
                class_name,
                file_path
            )
        except Exception as e:
            print(f"Файл {file_path} повреждён или не может быть загружен: {e}")
            return None

def collate_fn(batch):
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        return None, None, None
    fixed_num_points = 4096
    padded_batch = []
    faces_batch = []
    classes_batch = []
    for item in valid_items:
        vertices, faces, class_name, file_path = item
        current_num_points = vertices.size(0)
        if current_num_points >= fixed_num_points:
            padded_item = vertices[:fixed_num_points]
        else:
            padding = torch.zeros((fixed_num_points - current_num_points, 3))
            padded_item = torch.cat((vertices, padding), dim=0)
        padded_batch.append(padded_item)
        faces_batch.append(faces)
        classes_batch.append(class_name)
    return torch.stack(padded_batch), faces_batch, classes_batch
