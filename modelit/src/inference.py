import os
import argparse
import torch
import yaml
from types import SimpleNamespace
import trimesh

from models.text_to_3d_model import TextTo3DModel
from utils.voxelization import process_voxel_prediction, save_mesh

def load_config(config_path):
    """
    Загружает конфигурацию из YAML файла.
    
    Args:
        config_path (str): Путь к файлу конфигурации.
        
    Returns:
        SimpleNamespace: Объект с параметрами конфигурации.
    """
    with open(config_path, 'r') as f:
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

def load_model(config, checkpoint_path):
    """
    Загружает модель из чекпоинта.
    
    Args:
        config: Конфигурация модели.
        checkpoint_path (str): Путь к чекпоинту.
        
    Returns:
        TextTo3DModel: Загруженная модель.
    """
    # Инициализация модели
    model = TextTo3DModel(config)
    
    # Загрузка весов
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Переключение в режим оценки
    model.eval()
    
    return model

def generate_3d_model(model, text_prompt, device, threshold=0.5, smooth=True, sigma=0.5):
    """
    Генерирует 3D модель по текстовому описанию.
    
    Args:
        model: Модель для генерации.
        text_prompt (str): Текстовое описание.
        device: Устройство для вычислений.
        threshold (float): Порог бинаризации воксельной сетки.
        smooth (bool): Применять ли сглаживание.
        sigma (float): Параметр сглаживания Гаусса.
        
    Returns:
        trimesh.Trimesh: 3D меш.
    """
    # Перенос модели на нужное устройство
    model = model.to(device)
    
    with torch.no_grad():
        # Генерация воксельной сетки
        voxel_grid = model.generate(text_prompt, threshold)
        
        # Преобразование в numpy массив
        voxel_array = voxel_grid[0, 0].cpu().numpy()
        
        # Обработка воксельной сетки и преобразование в меш
        mesh = process_voxel_prediction(voxel_array, threshold, smooth, sigma)
    
    return mesh

def main():
    """
    Основная функция.
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Генерация 3D моделей из текстовых описаний")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Путь к чекпоинту модели")
    parser.add_argument("--prompt", type=str, required=True, help="Текстовое описание для генерации 3D модели")
    parser.add_argument("--output", type=str, default="models/output.obj", help="Путь для сохранения сгенерированной модели")
    parser.add_argument("--threshold", type=float, default=0.5, help="Порог бинаризации воксельной сетки")
    parser.add_argument("--smooth", action="store_true", help="Применять сглаживание к результату")
    parser.add_argument("--sigma", type=float, default=0.5, help="Параметр сглаживания Гаусса")
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Определение устройства для вычислений
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    print(f"Загрузка модели из {args.checkpoint}...")
    model = load_model(config, args.checkpoint)
    
    # Генерация 3D модели
    print(f"Генерация 3D модели для промпта: '{args.prompt}'")
    mesh = generate_3d_model(
        model, 
        args.prompt, 
        device, 
        threshold=args.threshold, 
        smooth=args.smooth, 
        sigma=args.sigma
    )
    
    # Проверка успешности генерации
    if mesh is None:
        print("Ошибка: не удалось сгенерировать 3D модель")
        return
    
    # Создание директории для сохранения, если она не существует
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Сохранение модели
    save_mesh(mesh, args.output)
    print(f"3D модель сохранена в {args.output}")
    
    # Вывод информации о модели
    print(f"Информация о модели:")
    print(f"  Количество вершин: {len(mesh.vertices)}")
    print(f"  Количество граней: {len(mesh.faces)}")

if __name__ == "__main__":
    main() 