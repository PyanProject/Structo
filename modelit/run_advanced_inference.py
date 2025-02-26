import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import sys
import trimesh
from scipy import ndimage

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.text_to_3d_model import TextTo3DModel
from src.utils.voxelization import save_mesh

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

def voxel_to_mesh_safe(voxels, threshold=0.5):
    """
    Безопасное преобразование воксельной сетки в меш с подробными сообщениями.
    """
    # Бинаризация воксельной сетки
    binary_voxels = voxels > threshold
    print(f"Бинаризация с порогом {threshold}: активных вокселей {np.sum(binary_voxels)} из {binary_voxels.size}")
    
    # Проверка на пустую сетку
    if np.sum(binary_voxels) == 0:
        print("Предупреждение: воксельная сетка пуста после бинаризации. Возвращаем базовый куб.")
        return trimesh.creation.box(extents=[1, 1, 1])
    
    # Попытка создать меш с использованием marching cubes
    try:
        # Сглаживание для лучшего результата
        smoothed_voxels = ndimage.gaussian_filter(voxels, sigma=0.5)
        
        # Создание меша с использованием марширующих кубов из scikit-image
        from skimage import measure
        vertices, faces, normals, values = measure.marching_cubes(smoothed_voxels, level=threshold)
        
        # Создание меша из trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        print(f"Успешно создан меш с использованием marching cubes: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
        
        return mesh
    
    except Exception as e:
        print(f"Ошибка при создании меша с использованием marching cubes: {str(e)}")
    
    # Запасной вариант: создание кубов для каждого активного вокселя
    try:
        print("Использование альтернативного метода: создание отдельных кубов для активных вокселей")
        
        # Находим координаты активных вокселей
        voxel_coords = np.where(binary_voxels > 0)
        
        if len(voxel_coords[0]) == 0:
            print("Предупреждение: воксельная сетка не содержит активных вокселей. Возвращаем одиночный куб.")
            return trimesh.creation.box(extents=[1, 1, 1])
        
        meshes = []
        
        # Ограничиваем количество кубов для оптимизации времени выполнения
        max_cubes = 1000
        coords_count = min(len(voxel_coords[0]), max_cubes)
        
        print(f"Создание {coords_count} кубов для активных вокселей...")
        
        for i in range(coords_count):
            # Создаем куб для каждого вокселя
            box = trimesh.creation.box(extents=[1, 1, 1])
            box.apply_translation([voxel_coords[0][i], voxel_coords[1][i], voxel_coords[2][i]])
            meshes.append(box)
        
        # Объединяем все кубы в один меш
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
            print(f"Успешно создан меш из {len(meshes)} кубов: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
            return mesh
        else:
            print("Предупреждение: не удалось создать ни одного куба. Возвращаем базовый куб.")
            return trimesh.creation.box(extents=[1, 1, 1])
            
    except Exception as e:
        print(f"Ошибка при создании меша из кубов: {str(e)}")
        print("Возвращаем базовый куб как крайний запасной вариант.")
        return trimesh.creation.box(extents=[1, 1, 1])

def visualize_voxel_grid(voxel_grid, output_path, threshold=0.5):
    """Визуализирует воксельную сетку с помощью срезов."""
    
    # Получаем размеры сетки
    depth, height, width = voxel_grid.shape
    
    # Создаем фигуру с тремя видами (срезами по основным осям)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Срез по оси X (средний срез)
    middle_x = depth // 2
    axes[0].imshow(voxel_grid[middle_x, :, :], cmap='viridis')
    axes[0].set_title(f'Срез по X (x={middle_x})')
    
    # Срез по оси Y (средний срез)
    middle_y = height // 2
    axes[1].imshow(voxel_grid[:, middle_y, :], cmap='viridis')
    axes[1].set_title(f'Срез по Y (y={middle_y})')
    
    # Срез по оси Z (средний срез)
    middle_z = width // 2
    axes[2].imshow(voxel_grid[:, :, middle_z], cmap='viridis')
    axes[2].set_title(f'Срез по Z (z={middle_z})')
    
    # Добавим информацию о пороге и статистике
    binary_grid = voxel_grid > threshold
    active_voxels = np.sum(binary_grid)
    total_voxels = binary_grid.size
    active_percent = (active_voxels / total_voxels) * 100
    
    plt.suptitle(f'Визуализация воксельной сетки\n'
                f'Порог: {threshold}, Активных вокселей: {active_voxels}/{total_voxels} ({active_percent:.2f}%)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Визуализация сохранена в {output_path}")

def generate_model(model, text_prompt, output_path, threshold=0.2, device="cpu"):
    """Генерирует 3D модель по текстовому описанию."""
    print(f"Генерация 3D модели для промпта: '{text_prompt}'")
    with torch.no_grad():
        try:
            # Генерация воксельной сетки
            print("Запрос модели для генерации воксельной сетки...")
            voxel_grid = model(text_prompt)
            print(f"Получен тензор формы {voxel_grid.shape}")
            
            # Преобразование в numpy массив и применение сигмоиды
            raw_voxel_array = voxel_grid[0, 0].detach().cpu().numpy()
            voxel_array = torch.sigmoid(voxel_grid[0, 0]).detach().cpu().numpy()
            
            print(f"Сырая воксельная сетка (до сигмоиды):")
            print(f"  Форма: {raw_voxel_array.shape}")
            print(f"  Мин/Макс: {raw_voxel_array.min():.4f}/{raw_voxel_array.max():.4f}")
            print(f"  Среднее/Медиана: {raw_voxel_array.mean():.4f}/{np.median(raw_voxel_array):.4f}")
            
            print(f"Воксельная сетка после сигмоиды:")
            print(f"  Форма: {voxel_array.shape}")
            print(f"  Мин/Макс: {voxel_array.min():.4f}/{voxel_array.max():.4f}")
            print(f"  Среднее/Медиана: {voxel_array.mean():.4f}/{np.median(voxel_array):.4f}")
            
            # Создание директорий для сохранения результатов
            vis_dir = os.path.join(os.path.dirname(output_path), "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Визуализация воксельной сетки с разными порогами
            for vis_threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                vis_path = os.path.join(vis_dir, f"{os.path.basename(output_path).split('.')[0]}_threshold_{vis_threshold:.1f}.png")
                visualize_voxel_grid(voxel_array, vis_path, threshold=vis_threshold)
            
            # Создание и сохранение меша
            print(f"Создание меша с порогом {threshold}...")
            mesh = voxel_to_mesh_safe(voxel_array, threshold=threshold)
            
            # Сохранение модели
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_mesh(mesh, output_path)
            print(f"3D модель сохранена в {output_path}")
            
            # Вывод информации о модели
            print(f"Информация о модели:")
            print(f"  Количество вершин: {len(mesh.vertices)}")
            print(f"  Количество граней: {len(mesh.faces)}")
            
            return mesh, voxel_array
        except Exception as e:
            print(f"Ошибка при генерации модели: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

if __name__ == "__main__":
    # Параметры по умолчанию
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs/train_config.yaml")
    checkpoint_path = os.path.join(os.path.dirname(base_dir), "checkpoints/best_model.pt")
    output_dir = os.path.join(os.path.dirname(base_dir), "models")
    
    # Проверка наличия аргументов командной строки
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="Генерация 3D моделей из текстовых описаний")
        parser.add_argument("--config", type=str, default=config_path, help="Путь к файлу конфигурации")
        parser.add_argument("--checkpoint", type=str, default=checkpoint_path, help="Путь к чекпоинту модели")
        parser.add_argument("--prompt", type=str, required=False, help="Текстовое описание для генерации 3D модели")
        parser.add_argument("--output", type=str, default=os.path.join(output_dir, "output.obj"), help="Путь для сохранения сгенерированной модели")
        parser.add_argument("--threshold", type=float, default=0.2, help="Порог бинаризации воксельной сетки")
        args = parser.parse_args()
        
        config_path = args.config
        checkpoint_path = args.checkpoint
        prompt = args.prompt
        output_path = args.output
        threshold = args.threshold
    else:
        # Значения по умолчанию при запуске без аргументов
        prompt = "3D model of an airplane"
        output_path = os.path.join(output_dir, "advanced_airplane.obj")
        threshold = 0.2

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
        
        # Вывод информации об эпохе и метриках из чекпоинта
        if 'epoch' in checkpoint:
            print(f"Чекпоинт сохранен на эпохе: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print("Метрики при сохранении:")
            for metric_name, metric_value in checkpoint['metrics'].items():
                print(f"  {metric_name}: {metric_value}")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {str(e)}")
        sys.exit(1)

    # Переключение в режим оценки и перенос на нужное устройство
    model.eval()
    model = model.to(device)

    # Генерация модели для заданного промпта
    mesh, voxel_array = generate_model(model, prompt, output_path, threshold, device)
    
    # Генерация с разными порогами для сравнения
    print("\nГенерация моделей с разными порогами для сравнения:")
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for thresh in thresholds:
        output_path_thresh = os.path.join(output_dir, f"threshold_{thresh:.2f}_{os.path.basename(output_path)}")
        print(f"\nПопытка создания модели с порогом {thresh}...")
        generate_model(model, prompt, output_path_thresh, thresh, device)
    
    print("\nГенерация завершена.") 