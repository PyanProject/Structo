#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для загрузки и подготовки данных из датасета Objaverse.
"""

import os
import json
import argparse
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

try:
    import objaverse
except ImportError:
    logging.error("Библиотека objaverse не установлена. Используйте 'pip install objaverse'")
    raise

try:
    import tqdm
except ImportError:
    tqdm = None

# Настройка логирования
def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логирования.
    
    Args:
        log_file: Путь к файлу логов.
        level: Уровень логирования.
        
    Returns:
        Настроенный логгер.
    """
    logger = logging.getLogger("data_download")
    logger.setLevel(level)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый обработчик, если указан файл
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Загрузка данных из Objaverse
def download_objaverse(
    limit: int = 1000,
    download_dir: str = "data/objaverse/raw",
    random_seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Загрузка объектов из Objaverse.
    
    Args:
        limit: Максимальное количество объектов для загрузки.
        download_dir: Директория для сохранения загруженных объектов.
        random_seed: Зерно для случайного выбора объектов.
        logger: Логгер для вывода сообщений.
        
    Returns:
        Список идентификаторов загруженных объектов.
    """
    if logger is None:
        logger = logging.getLogger("data_download")
    
    download_dir = Path(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    
    logger.info(f"Начинаю загрузку данных из Objaverse (лимит: {limit} объектов)")
    
    # Получаем список доступных объектов
    logger.info("Получение списка доступных объектов...")
    random.seed(random_seed)
    
    if limit > 0:
        # Загружаем только часть объектов
        try:
            # Получаем все идентификаторы объектов
            all_object_ids = objaverse.load_uids()
            logger.info(f"Доступно {len(all_object_ids)} объектов")
            
            # Выбираем случайные объекты
            selected_ids = random.sample(all_object_ids, min(limit, len(all_object_ids)))
            logger.info(f"Выбрано {len(selected_ids)} объектов для загрузки")
        except Exception as e:
            logger.error(f"Ошибка при получении списка объектов: {e}")
            return []
    else:
        # Загружаем все объекты
        try:
            selected_ids = objaverse.load_uids()
            logger.info(f"Получено {len(selected_ids)} идентификаторов объектов")
        except Exception as e:
            logger.error(f"Ошибка при получении списка объектов: {e}")
            return []
    
    # Загружаем объекты
    logger.info(f"Начинаю загрузку {len(selected_ids)} объектов...")
    start_time = time.time()
    
    try:
        # Загружаем объекты с прогрессбаром, если доступен tqdm
        if tqdm:
            objects = objaverse.load_objects(
                uids=selected_ids,
                download_dir=str(download_dir),
                show_progress=True
            )
        else:
            objects = objaverse.load_objects(
                uids=selected_ids,
                download_dir=str(download_dir),
                show_progress=False
            )
        
        logger.info(f"Загружено {len(objects)} объектов за {time.time() - start_time:.2f} секунд")
        return list(objects.keys())
    except Exception as e:
        logger.error(f"Ошибка при загрузке объектов: {e}")
        return []

# Создание метаданных
def create_metadata(
    object_ids: List[str],
    download_dir: str = "data/objaverse/raw",
    output_file: str = "data/objaverse/metadata.json",
    captions_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict]:
    """
    Создание файла метаданных для загруженных объектов.
    
    Args:
        object_ids: Список идентификаторов объектов.
        download_dir: Директория с загруженными объектами.
        output_file: Путь для сохранения файла метаданных.
        captions_path: Путь к файлу с описаниями объектов (если есть).
        logger: Логгер для вывода сообщений.
        
    Returns:
        Словарь с метаданными объектов.
    """
    if logger is None:
        logger = logging.getLogger("data_download")
    
    download_dir = Path(download_dir)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    logger.info(f"Создание метаданных для {len(object_ids)} объектов")
    
    # Загружаем описания, если указан путь
    captions = {}
    if captions_path and os.path.exists(captions_path):
        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                captions = json.load(f)
            logger.info(f"Загружено {len(captions)} описаний объектов")
        except Exception as e:
            logger.error(f"Ошибка при загрузке описаний: {e}")
    
    # Создаем метаданные
    metadata = {}
    valid_count = 0
    
    for obj_id in object_ids:
        # Проверяем наличие директории с объектом
        obj_dir = download_dir / obj_id
        if not obj_dir.exists():
            logger.warning(f"Директория для объекта {obj_id} не найдена")
            continue
        
        # Проверяем наличие файла модели
        model_files = list(obj_dir.glob("*.obj")) + list(obj_dir.glob("*.glb")) + list(obj_dir.glob("*.ply"))
        if not model_files:
            logger.warning(f"Файл модели для объекта {obj_id} не найден")
            continue
        
        # Формируем метаданные
        metadata[obj_id] = {
            "id": obj_id,
            "path": str(obj_dir.relative_to(download_dir.parent)),
            "caption": captions.get(obj_id, f"3D model with ID {obj_id}"),
            "file_format": model_files[0].suffix[1:],
            "timestamp": datetime.now().isoformat()
        }
        valid_count += 1
    
    # Сохраняем метаданные
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Создано {valid_count} записей метаданных, сохранено в {output_file}")
    return metadata

# Создание разбиений
def create_splits(
    object_ids: List[str],
    output_dir: str = "data/objaverse",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Создание разбиений на train/val/test.
    
    Args:
        object_ids: Список идентификаторов объектов.
        output_dir: Директория для сохранения файлов разбиений.
        train_ratio: Доля объектов для обучения.
        val_ratio: Доля объектов для валидации.
        random_seed: Зерно для случайного перемешивания.
        logger: Логгер для вывода сообщений.
        
    Returns:
        Кортеж со списками идентификаторов объектов для каждого разбиения.
    """
    if logger is None:
        logger = logging.getLogger("data_download")
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Создание разбиений для {len(object_ids)} объектов")
    
    # Перемешиваем идентификаторы
    random.seed(random_seed)
    shuffled_ids = object_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Вычисляем размеры разбиений
    train_size = int(len(shuffled_ids) * train_ratio)
    val_size = int(len(shuffled_ids) * val_ratio)
    
    # Создаем разбиения
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:train_size+val_size]
    test_ids = shuffled_ids[train_size+val_size:]
    
    # Сохраняем разбиения
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(train_ids))
    
    with open(output_dir / "val.txt", "w") as f:
        f.write("\n".join(val_ids))
    
    with open(output_dir / "test.txt", "w") as f:
        f.write("\n".join(test_ids))
    
    logger.info(f"Создано разбиение: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return train_ids, val_ids, test_ids

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Загрузка и подготовка данных из Objaverse")
    parser.add_argument("--limit", type=int, default=1000, help="Максимальное количество объектов для загрузки")
    parser.add_argument("--download_dir", type=str, default="data/objaverse/raw", help="Директория для сохранения загруженных объектов")
    parser.add_argument("--output_dir", type=str, default="data/objaverse", help="Директория для сохранения метаданных и разбиений")
    parser.add_argument("--captions_path", type=str, default=None, help="Путь к файлу с описаниями объектов")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Доля объектов для обучения")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Доля объектов для валидации")
    parser.add_argument("--random_seed", type=int, default=42, help="Зерно для случайного выбора объектов")
    parser.add_argument("--log_file", type=str, default=None, help="Путь к файлу логов")
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(args.log_file)
    
    # Создаем корневую директорию, если она не существует
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем объекты
    object_ids = download_objaverse(
        limit=args.limit,
        download_dir=args.download_dir,
        random_seed=args.random_seed,
        logger=logger
    )
    
    if not object_ids:
        logger.error("Не удалось загрузить объекты")
        return
    
    # Создаем метаданные
    metadata = create_metadata(
        object_ids=object_ids,
        download_dir=args.download_dir,
        output_file=os.path.join(args.output_dir, "metadata.json"),
        captions_path=args.captions_path,
        logger=logger
    )
    
    # Создаем разбиения
    train_ids, val_ids, test_ids = create_splits(
        object_ids=list(metadata.keys()),
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        logger=logger
    )
    
    logger.info("Обработка данных завершена")

if __name__ == "__main__":
    main() 