import os
import torch
import yaml
import numpy as np
from types import SimpleNamespace
import sys
import trimesh

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.text_to_3d_model import TextTo3DModel
from src.utils.voxelization import process_voxel_prediction, save_mesh, voxel_to_mesh

def load_config(config_path):
    """Загружает конфигурацию из YAML файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Преобразование словаря в объект с атрибутами
    config = SimpleNamespace(**config_dict)
    
    # Преобразование вложенных словарей в объекты
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(config, key, SimpleNamespace(**value))
            
            # Обработка вложенных словарей второго уровня
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    setattr(getattr(config, key), subkey, SimpleNamespace(**subvalue))
    
    return config

def generate_model(model, text_prompt, output_path, threshold=0.2, device="cpu"):
    """Генерирует 3D модель по текстовому описанию."""
    print(f"Генерация 3D модели для промпта: '{text_prompt}'")
    with torch.no_grad():
        try:
            # Генерация воксельной сетки
            voxel_grid = model(text_prompt)
            
            # Преобразование в numpy массив и применение сигмоиды
            voxel_array = torch.sigmoid(voxel_grid[0, 0]).detach().cpu().numpy()
            
            # Бинаризация воксельной сетки
            binary_voxels = (voxel_array > threshold).astype(np.float32)
            
            print(f"Форма воксельной сетки: {binary_voxels.shape}")
            print(f"Минимальное значение: {binary_voxels.min()}, Максимальное значение: {binary_voxels.max()}")
            print(f"Количество ненулевых вокселей: {np.count_nonzero(binary_voxels)}")
            
            # Попытка создать меш из бинарной воксельной сетки
            try:
                print("Попытка создания меша с помощью функции voxel_to_mesh...")
                mesh = voxel_to_mesh(binary_voxels)
            except Exception as e:
                print(f"Ошибка при создании меша из воксельной сетки: {str(e)}")
                
                # Альтернативный метод: создание кубиков для каждого ненулевого вокселя
                print("Использование альтернативного метода создания меша...")
                voxels = np.where(binary_voxels > 0)
                
                if len(voxels[0]) == 0:
                    print("Воксельная сетка пуста (нет значений больше порога)")
                    # Создаем простой куб, чтобы хоть что-то вернуть
                    mesh = trimesh.creation.box(extents=[1, 1, 1])
                else:
                    meshes = []
                    for i, j, k in zip(voxels[0], voxels[1], voxels[2]):
                        # Создаем куб для каждого вокселя
                        box = trimesh.creation.box(extents=[1, 1, 1])
                        box.apply_translation([i, j, k])
                        meshes.append(box)
                    
                    if meshes:
                        # Объединяем все кубы в один меш
                        mesh = trimesh.util.concatenate(meshes)
                    else:
                        # Если не удалось создать ни одного куба
                        mesh = trimesh.creation.box(extents=[1, 1, 1])
            
            # Создание директории для сохранения, если она не существует
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохранение модели
            save_mesh(mesh, output_path)
            print(f"3D модель сохранена в {output_path}")
            
            # Вывод информации о модели
            print(f"Информация о модели:")
            print(f"  Количество вершин: {len(mesh.vertices)}")
            print(f"  Количество граней: {len(mesh.faces)}")
            
            return mesh
        except Exception as e:
            print(f"Ошибка при генерации модели: {str(e)}")
            return None

# Параметры по умолчанию
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "configs/train_config.yaml")
checkpoint_path = os.path.join(os.path.dirname(base_dir), "checkpoints/best_model.pt")
output_dir = os.path.join(os.path.dirname(base_dir), "models")

# Загрузка конфигурации
print(f"Загрузка конфигурации из {config_path}...")
config = load_config(config_path)

# Определение устройства для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Инициализация модели
print(f"Инициализация модели...")
model = TextTo3DModel(config)

# Загрузка весов
print(f"Загрузка весов из {checkpoint_path}...")
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Веса модели успешно загружены")
except Exception as e:
    print(f"Ошибка при загрузке весов: {str(e)}")
    sys.exit(1)

# Переключение в режим оценки и перенос на нужное устройство
model.eval()
model = model.to(device)

# Генерация моделей для различных промптов
prompts = [
    "3D model of an airplane",
    "3D model of chair",
    "3D model of table",
    "3D model of cup"
]

# Различные пороги бинаризации
thresholds = [0.3, 0.2, 0.1, 0.05]

for i, prompt in enumerate(prompts):
    for j, threshold in enumerate(thresholds):
        output_path = os.path.join(output_dir, f"generated_{i}_{j}_thresh_{threshold:.2f}.obj")
        generate_model(model, prompt, output_path, threshold, device) 