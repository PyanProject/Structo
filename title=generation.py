# Dataset for потоковой загрузки Objaverse-XL объектов без полного скачивания.
import objaverse
class ObjaverseVoxelDataset(Dataset):
    def __init__(self, split="train", voxel_size=64, limit=None):
        self.voxel_size = voxel_size
        # Загружаем UID всех объектов
        self.uids = objaverse.load_uids()
        if limit is not None:
            self.uids = self.uids[:limit]

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        # Загружаем выбранный объект по UID. Функция возвращает словарь, где ключ – UID,
        # а значение – путь к 3D-файлу (либо URL, либо локальный путь).
        objs = objaverse.load_objects([uid])
        file_path = objs[uid]
        mesh = trimesh.load(file_path, force='mesh')
        pitch = mesh.bounding_box.extents.max() / self.voxel_size
        voxel_obj = mesh.voxelized(pitch)
        voxels = voxel_obj.matrix.astype(np.float32)
        grid = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=voxels.dtype)
        v_shape = voxels.shape
        slices_voxels = []
        slices_grid = []
        for d, s in enumerate(v_shape):
            if s <= self.voxel_size:
                grid_start = (self.voxel_size - s) // 2
                grid_end = grid_start + s
                vox_start = 0
                vox_end = s
            else:
                vox_start = (s - self.voxel_size) // 2
                vox_end = vox_start + self.voxel_size
                grid_start = 0
                grid_end = self.voxel_size
            slices_voxels.append(slice(vox_start, vox_end))
            slices_grid.append(slice(grid_start, grid_end))
        grid[slices_grid[0], slices_grid[1], slices_grid[2]] = voxels[slices_voxels[0], slices_voxels[1], slices_voxels[2]]
        voxel_tensor = torch.tensor(grid).unsqueeze(0)
        return voxel_tensor, "" 