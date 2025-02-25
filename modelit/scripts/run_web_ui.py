import os
import sys
import argparse
import torch
import gradio as gr
import yaml
from types import SimpleNamespace
import trimesh
import numpy as np
import tempfile

# Добавление корневой директории проекта в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.text_to_3d_model import TextTo3DModel
from src.utils.voxelization import process_voxel_prediction, save_mesh

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

def load_model(config, checkpoint_path, device):
    """
    Загружает модель из чекпоинта.
    
    Args:
        config: Конфигурация модели.
        checkpoint_path (str): Путь к чекпоинту.
        device: Устройство для вычислений.
        
    Returns:
        TextTo3DModel: Загруженная модель.
    """
    # Инициализация модели
    model = TextTo3DModel(config)
    
    # Загрузка весов
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Переключение в режим оценки
    model.eval()
    
    # Перенос модели на нужное устройство
    model = model.to(device)
    
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
    with torch.no_grad():
        # Генерация воксельной сетки
        voxel_grid = model.generate(text_prompt, threshold)
        
        # Преобразование в numpy массив
        voxel_array = voxel_grid[0, 0].cpu().numpy()
        
        # Обработка воксельной сетки и преобразование в меш
        mesh = process_voxel_prediction(voxel_array, threshold, smooth, sigma)
    
    return mesh

def create_web_ui(model, device):
    """
    Создает веб-интерфейс для генерации 3D моделей.
    
    Args:
        model: Модель для генерации.
        device: Устройство для вычислений.
        
    Returns:
        gr.Interface: Интерфейс Gradio.
    """
    def generate_model(text_prompt, threshold, smooth, sigma, output_format):
        """
        Функция для генерации 3D модели через веб-интерфейс.
        
        Args:
            text_prompt (str): Текстовое описание.
            threshold (float): Порог бинаризации воксельной сетки.
            smooth (bool): Применять ли сглаживание.
            sigma (float): Параметр сглаживания Гаусса.
            output_format (str): Формат выходного файла.
            
        Returns:
            str: Путь к сгенерированной модели.
        """
        # Проверка входных данных
        if not text_prompt.strip():
            return None, "Ошибка: текстовое описание не может быть пустым"
        
        try:
            # Генерация 3D модели
            mesh = generate_3d_model(model, text_prompt, device, threshold, smooth, sigma)
            
            if mesh is None:
                return None, "Ошибка: не удалось сгенерировать 3D модель"
            
            # Создание временного файла для сохранения модели
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
                tmp_path = tmp.name
            
            # Сохранение модели
            save_mesh(mesh, tmp_path)
            
            # Информация о модели
            info = f"Сгенерирована 3D модель по описанию: '{text_prompt}'\n"
            info += f"Количество вершин: {len(mesh.vertices)}\n"
            info += f"Количество граней: {len(mesh.faces)}\n"
            info += f"Формат: {output_format}"
            
            return tmp_path, info
            
        except Exception as e:
            return None, f"Ошибка при генерации модели: {str(e)}"
    
    # Создание интерфейса
    interface = gr.Interface(
        fn=generate_model,
        inputs=[
            gr.Textbox(label="Текстовое описание", placeholder="Введите описание 3D объекта, например: 'современный стул с изогнутой спинкой'"),
            gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Порог бинаризации"),
            gr.Checkbox(label="Применять сглаживание", value=True),
            gr.Slider(minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="Параметр сглаживания"),
            gr.Dropdown(["obj", "stl", "ply"], value="obj", label="Формат выходного файла")
        ],
        outputs=[
            gr.Model3D(label="Сгенерированная 3D модель"),
            gr.Textbox(label="Информация")
        ],
        title="ModelIt: Генерация 3D моделей из текстовых описаний",
        description="Введите текстовое описание объекта, и модель сгенерирует соответствующую 3D модель.",
        examples=[
            ["современный стул с изогнутой спинкой", 0.5, True, 0.5, "obj"],
            ["круглый стол с четырьмя ножками", 0.5, True, 0.5, "obj"],
            ["простая ваза цилиндрической формы", 0.5, True, 0.5, "obj"],
            ["компьютерная клавиатура", 0.5, True, 0.5, "obj"],
            ["спортивный автомобиль", 0.5, True, 0.5, "obj"]
        ]
    )
    
    return interface

def main():
    """
    Основная функция.
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Запуск веб-интерфейса для генерации 3D моделей")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Путь к чекпоинту модели")
    parser.add_argument("--port", type=int, default=7860, help="Порт для запуска веб-интерфейса")
    parser.add_argument("--share", action="store_true", help="Сделать интерфейс доступным публично")
    args = parser.parse_args()
    
    # Проверка наличия чекпоинта
    if not os.path.exists(args.checkpoint):
        print(f"Ошибка: чекпоинт {args.checkpoint} не найден")
        print("Пожалуйста, сначала обучите модель или укажите путь к существующему чекпоинту")
        return
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Определение устройства для вычислений
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    print(f"Загрузка модели из {args.checkpoint}...")
    model = load_model(config, args.checkpoint, device)
    print("Модель успешно загружена")
    
    # Создание и запуск веб-интерфейса
    print(f"Запуск веб-интерфейса на порту {args.port}...")
    interface = create_web_ui(model, device)
    interface.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    main() 