import torch
import random
import numpy as np
import os
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    Устанавливает seed для воспроизводимости результатов.
    
    Args:
        seed (int): Начальное значение для генератора случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Установлен seed {seed} для воспроизводимости результатов")

def setup_logging(log_dir, experiment_name):
    """
    Настраивает систему логирования.
    
    Args:
        log_dir (str): Директория для сохранения логов.
        experiment_name (str): Название эксперимента.
        
    Returns:
        str: Путь к директории с логами текущего запуска.
    """
    # Создаем директорию для логов, если она не существует
    os.makedirs(log_dir, exist_ok=True)
    
    # Создаем поддиректорию для текущего запуска
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Настройка форматирования логов
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Логи сохраняются в {run_log_dir}")
    
    return run_log_dir

def save_config(config, save_path):
    """
    Сохраняет конфигурацию в JSON-файл.
    
    Args:
        config: Объект конфигурации.
        save_path (str): Путь для сохранения.
    """
    # Преобразование конфигурации в словарь
    if hasattr(config, "__dict__"):
        config_dict = {}
        for key, value in vars(config).items():
            if hasattr(value, "__dict__"):
                config_dict[key] = vars(value)
            else:
                config_dict[key] = value
    else:
        config_dict = config
    
    # Сохранение в JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Конфигурация сохранена в {save_path}")

def get_lr(optimizer):
    """
    Возвращает текущую скорость обучения оптимизатора.
    
    Args:
        optimizer: Оптимизатор.
        
    Returns:
        float: Текущая скорость обучения.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_parameters(model):
    """
    Подсчитывает количество параметров модели.
    
    Args:
        model: Модель.
        
    Returns:
        tuple: (всего параметров, обучаемых параметров)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Всего параметров: {total_params:,}")
    logger.info(f"Обучаемых параметров: {trainable_params:,}")
    logger.info(f"Необучаемых параметров: {total_params - trainable_params:,}")
    
    return total_params, trainable_params

def format_time(seconds):
    """
    Форматирует время в удобочитаемый вид.
    
    Args:
        seconds (float): Время в секундах.
        
    Returns:
        str: Отформатированное время.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pt"):
    """
    Сохраняет чекпоинт модели.
    
    Args:
        state (dict): Состояние для сохранения.
        is_best (bool): Является ли текущая модель лучшей.
        checkpoint_dir (str): Директория для сохранения чекпоинтов.
        filename (str): Имя файла чекпоинта.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Сохранение чекпоинта
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    logger.info(f"Чекпоинт сохранен в {checkpoint_path}")
    
    # Если это лучшая модель, сохраняем копию
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_path)
        logger.info(f"Лучшая модель сохранена в {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Загружает чекпоинт модели.
    
    Args:
        checkpoint_path (str): Путь к чекпоинту.
        model: Модель для загрузки весов.
        optimizer: Оптимизатор для загрузки состояния.
        
    Returns:
        dict: Дополнительная информация из чекпоинта.
    """
    if not os.path.isfile(checkpoint_path):
        logger.error(f"Чекпоинт не найден: {checkpoint_path}")
        return {}
    
    # Загрузка чекпоинта
    checkpoint = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
    
    # Загрузка весов модели
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Загрузка состояния оптимизатора, если он передан
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Чекпоинт загружен из {checkpoint_path}")
    
    # Возвращаем дополнительную информацию
    info = {}
    for key in checkpoint:
        if key not in ['model_state_dict', 'optimizer_state_dict']:
            info[key] = checkpoint[key]
    
    return info

def create_experiment_dir(base_dir, experiment_name):
    """
    Создает директорию для эксперимента с уникальным именем.
    
    Args:
        base_dir (str): Базовая директория.
        experiment_name (str): Название эксперимента.
        
    Returns:
        str: Путь к директории эксперимента.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger.info(f"Создана директория эксперимента: {experiment_dir}")
    
    return experiment_dir 