import os
import torch
import yaml
import numpy as np
from types import SimpleNamespace
import sys
import trimesh
from skimage import measure
from scipy import ndimage

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.text_to_3d_model import TextTo3DModel

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

def save_mesh(mesh, path):
    """Сохраняет mesh в файл."""
    try:
        mesh.export(path)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении меша: {str(e)}")
        return False

def voxel_to_mesh(voxels, threshold=0.5):
    """
    Преобразует воксельную сетку в меш с помощью marching cubes.
    
    Args:
        voxels (numpy.ndarray): Воксельная сетка.
        threshold (float): Порог бинаризации.
        
    Returns:
        trimesh.Trimesh: Меш.
    """
    try:
        # Применяем сглаживание для лучшего результата
        smoothed_voxels = ndimage.gaussian_filter(voxels, sigma=0.5)
        
        # Бинаризация воксельной сетки
        binary_voxels = smoothed_voxels > threshold
        
        print(f"Активных вокселей после бинаризации: {np.sum(binary_voxels)} из {binary_voxels.size}")
        
        # Если нет активных вокселей, возвращаем простой куб
        if np.sum(binary_voxels) == 0:
            print("Предупреждение: воксельная сетка пуста после бинаризации.")
            return trimesh.creation.box(extents=[1, 1, 1])
        
        # Создание меша с использованием marching cubes
        vertices, faces, normals, _ = measure.marching_cubes(smoothed_voxels, level=threshold)
        
        # Создание меша из trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        
        print(f"Создан меш с {len(mesh.vertices)} вершинами и {len(mesh.faces)} гранями.")
        return mesh
        
    except Exception as e:
        print(f"Ошибка при создании меша: {str(e)}")
        
        # Запасной вариант: создаем кубы для каждого активного вокселя
        try:
            print("Использование альтернативного метода: создание кубов для каждого активного вокселя")
            
            # Находим координаты активных вокселей
            binary_voxels = voxels > threshold
            voxel_coords = np.where(binary_voxels)
            
            if len(voxel_coords[0]) == 0:
                print("Предупреждение: нет активных вокселей.")
                return trimesh.creation.box(extents=[1, 1, 1])
            
            print(f"Найдено {len(voxel_coords[0])} активных вокселей.")
            
            # Ограничиваем количество кубов для производительности
            max_cubes = 1000
            coords_count = min(len(voxel_coords[0]), max_cubes)
            
            meshes = []
            for i in range(coords_count):
                box = trimesh.creation.box(extents=[1, 1, 1])
                box.apply_translation([voxel_coords[0][i], voxel_coords[1][i], voxel_coords[2][i]])
                meshes.append(box)
            
            # Объединяем все кубы в один меш
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
                print(f"Создан меш из {len(meshes)} кубов: {len(mesh.vertices)} вершин и {len(mesh.faces)} граней.")
                return mesh
            else:
                return trimesh.creation.box(extents=[1, 1, 1])
                
        except Exception as e2:
            print(f"Ошибка при создании альтернативного меша: {str(e2)}")
            return trimesh.creation.box(extents=[1, 1, 1])

def visualize_voxel_stats(voxel_grid, threshold=0.5):
    """Выводит подробную статистику о воксельной сетке."""
    print(f"\nСтатистика воксельной сетки:")
    print(f"Форма: {voxel_grid.shape}")
    print(f"Минимум: {voxel_grid.min():.6f}, Максимум: {voxel_grid.max():.6f}")
    print(f"Среднее: {voxel_grid.mean():.6f}, Медиана: {np.median(voxel_grid):.6f}")
    
    # Гистограмма значений
    hist, bins = np.histogram(voxel_grid.flatten(), bins=10)
    print("\nГистограмма значений:")
    for i in range(len(hist)):
        print(f"  {bins[i]:.4f} - {bins[i+1]:.4f}: {hist[i]} значений")
    
    # Информация о бинаризации
    binary_grid = voxel_grid > threshold
    active_voxels = np.sum(binary_grid)
    total_voxels = binary_grid.size
    print(f"\nАктивных вокселей при пороге {threshold}: {active_voxels}/{total_voxels} ({active_voxels/total_voxels*100:.4f}%)")
    
    # Информация о верхних значениях
    top_values = np.sort(voxel_grid.flatten())[-10:]
    print(f"\nТоп-10 самых высоких значений:")
    for i, val in enumerate(reversed(top_values)):
        print(f"  {i+1}. {val:.6f}")

def generate_model(model, text_prompt, output_path, threshold=0.2, device="cpu"):
    """Генерирует 3D модель по текстовому описанию."""
    print(f"\nГенерация 3D модели для промпта: '{text_prompt}'")
    try:
        # Переводим модель в режим оценки
        model.eval()
        
        # Генерация воксельной сетки
        with torch.no_grad():
            voxel_grid = model(text_prompt)
            print(f"Получен тензор формы {voxel_grid.shape}")
            
            # Преобразование в numpy массив и применение сигмоиды
            raw_voxel_array = voxel_grid[0, 0].detach().cpu().numpy()
            voxel_array = torch.sigmoid(voxel_grid[0, 0]).detach().cpu().numpy()
            
            # Вывод статистики
            print(f"\nСырая воксельная сетка (до сигмоиды):")
            print(f"  Форма: {raw_voxel_array.shape}")
            print(f"  Мин/Макс: {raw_voxel_array.min():.4f}/{raw_voxel_array.max():.4f}")
            print(f"  Среднее/Медиана: {raw_voxel_array.mean():.4f}/{np.median(raw_voxel_array):.4f}")
            
            print(f"\nВоксельная сетка после сигмоиды:")
            print(f"  Форма: {voxel_array.shape}")
            print(f"  Мин/Макс: {voxel_array.min():.4f}/{voxel_array.max():.4f}")
            print(f"  Среднее/Медиана: {voxel_array.mean():.4f}/{np.median(voxel_array):.4f}")
            
            # Подробная статистика
            visualize_voxel_stats(voxel_array, threshold)
            
            # Создание меша
            print(f"\nСоздание меша с порогом {threshold}...")
            mesh = voxel_to_mesh(voxel_array, threshold=threshold)
            
            # Создание директории для сохранения
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохранение меша
            save_mesh(mesh, output_path)
            print(f"3D модель сохранена в {output_path}")
            
            # Вывод информации о модели
            print(f"Информация о сохраненной модели:")
            print(f"  Количество вершин: {len(mesh.vertices)}")
            print(f"  Количество граней: {len(mesh.faces)}")
            
            return mesh, voxel_array
            
    except Exception as e:
        print(f"Ошибка при генерации модели: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    # Параметры по умолчанию
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs/inference_config.yaml")
    checkpoint_path = os.path.join(os.path.dirname(base_dir), "checkpoints/best_model.pt")
    output_dir = os.path.join(os.path.dirname(base_dir), "models")
    
    # Проверка наличия аргументов командной строки
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="Генерация 3D моделей из текстовых описаний")
        parser.add_argument("--config", type=str, default=config_path, help="Путь к файлу конфигурации")
        parser.add_argument("--checkpoint", type=str, default=checkpoint_path, help="Путь к чекпоинту модели")
        parser.add_argument("--prompt", type=str, default="3D model of an airplane", help="Текстовое описание для генерации 3D модели")
        parser.add_argument("--output", type=str, default=os.path.join(output_dir, "fixed_output.obj"), help="Путь для сохранения сгенерированной модели")
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
        output_path = os.path.join(output_dir, "fixed_airplane.obj")
        threshold = 0.2
    
    print(f"Использование конфигурации: {config_path}")
    print(f"Загрузка модели из: {checkpoint_path}")
    print(f"Промпт для генерации: '{prompt}'")
    print(f"Сохранение результата в: {output_path}")
    print(f"Порог бинаризации: {threshold}")

    # Загрузка конфигурации
    try:
        config = load_config(config_path)
        print("Конфигурация успешно загружена")
    except Exception as e:
        print(f"Ошибка при загрузке конфигурации: {str(e)}")
        sys.exit(1)

    # Определение устройства для вычислений
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Инициализация модели
    try:
        print("Инициализация модели...")
        model = TextTo3DModel(config)
        print("Модель успешно инициализирована")
    except Exception as e:
        print(f"Ошибка при инициализации модели: {str(e)}")
        sys.exit(1)

    # Загрузка весов
    try:
        print(f"Загрузка весов из {checkpoint_path}...")
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

    # Перенос модели на устройство
    model = model.to(device)

    # Генерация модели для заданного промпта
    generate_model(model, prompt, output_path, threshold, device)
    
    # Генерация с разными порогами для сравнения
    print("\nГенерация моделей с разными порогами для сравнения:")
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for thresh in thresholds:
        output_path_thresh = os.path.join(output_dir, f"fixed_threshold_{thresh:.2f}_{os.path.basename(output_path)}")
        print(f"\nПопытка создания модели с порогом {thresh}...")
        generate_model(model, prompt, output_path_thresh, thresh, device)
    
    print("\nГенерация завершена.")

if __name__ == "__main__":
    main() 