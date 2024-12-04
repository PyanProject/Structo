import torch

# Пример создания тензора
voxels = torch.randn(10, 10, 10)  # 3D тензор

# Теперь можно использовать .view
voxels = voxels.view(-1, *voxels.shape)
