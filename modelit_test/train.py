#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import json
import yaml
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Оптимизация памяти PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импортируем наши модули
from src.objaverse_dataset import ObjaverseDataset, prepare_objaverse_data, get_objaverse_dataloader
from src.models.diffusion import GaussianDiffusion
from src.models.transformer import PointDiffusionTransformer, TextConditionedTransformer
from src.models.utils import load_config, load_model

def is_rocm_available():
    """Проверка доступности ROCm (для AMD GPU)"""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None

def get_device_info():
    """Получение информации о доступном устройстве"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"CUDA: {device_name} с {memory:.2f}GB памяти"
    elif is_rocm_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"ROCm: {device_name} с {memory:.2f}GB памяти"
    else:
        return "CPU (GPU не обнаружен)"

def parse_args():
    """Разбор аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Обучение нейросетевой модели для 3D генерации")
    
    # Основные параметры
    parser.add_argument("--data_dir", type=str, default="./data", help="Директория с данными")
    parser.add_argument("--output_dir", type=str, default="./models", help="Директория для сохранения моделей")
    parser.add_argument("--model_type", type=str, default="text300M", help="Тип модели: text300M или base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Устройство для обучения")
    
    # Параметры данных
    parser.add_argument("--prepare_data", action="store_true", help="Только подготовить данные без обучения")
    parser.add_argument("--num_objects", type=int, default=1000, help="Количество 3D объектов для подготовки")
    parser.add_argument("--num_workers", type=int, default=4, help="Количество рабочих процессов для загрузки данных")
    
    # Параметры обучения
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Шаги накопления градиента")
    parser.add_argument("--num_epochs", type=int, default=10, help="Количество эпох обучения")
    parser.add_argument("--val_freq", type=int, default=1, help="Частота валидации (в эпохах)")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    
    # Опции для экспериментальных моделей
    parser.add_argument("--config", type=str, default=None, help="Путь к файлу конфигурации YAML")
    parser.add_argument("--skip_pretrained", action="store_true", help="Пропустить загрузку предобученных весов")
    
    # Параметры модели, которые можно переопределить (гибкая настройка)
    parser.add_argument("--n_ctx", type=int, default=None, help="Максимальное количество точек")
    parser.add_argument("--width", type=int, default=None, help="Ширина скрытого слоя модели")
    parser.add_argument("--layers", type=int, default=None, help="Количество слоёв трансформера")
    parser.add_argument("--heads", type=int, default=None, help="Количество голов внимания")
    parser.add_argument("--dropout", type=float, default=None, help="Вероятность dropout")
    parser.add_argument("--text_embed_dim", type=int, default=None, help="Размерность текстовых эмбеддингов")
    
    # Автоматическое масштабирование модели
    parser.add_argument("--auto_scale", action="store_true", help="Автоматически масштабировать модель под доступную память GPU")
    
    args = parser.parse_args()
    return args

def validate_model(model, val_dataloader, device, diffusion):
    """Валидация модели на проверочном наборе"""
    model.eval()
    val_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            try:
                # Извлекаем данные
                point_cloud = batch['points'].to(device)
                text = batch.get('text', None)
                
                # Выбираем случайные моменты времени
                t = torch.randint(
                    0, diffusion.num_timesteps, 
                    (point_cloud.shape[0],), 
                    device=device
                ).long()
                
                # Готовим аргументы модели
                model_kwargs = {}
                if text is not None and hasattr(model, 'condition_on_text'):
                    model_kwargs['texts'] = text
                
                # Вычисляем потери
                with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    losses = diffusion.training_losses(model, point_cloud, t, model_kwargs=model_kwargs)
                
                val_loss += losses["loss"].item()
                n_batches += 1
                
                # Освобождаем память
                del point_cloud, t, losses
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Ошибка при валидации: {str(e)}")
                continue
    
    # Возвращаем среднюю потерю
    return {"loss": val_loss / max(1, n_batches)} if n_batches > 0 else {"loss": float('inf')}

def empty_cache_and_collect_garbage():
    """Освобождение памяти GPU и сбор мусора"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_scaling_profile(available_memory_gb):
    """
    Возвращает рекомендуемые параметры модели в зависимости от доступной памяти GPU
    
    Args:
        available_memory_gb (float): Доступная память GPU в гигабайтах
        
    Returns:
        dict: Словарь с рекомендуемыми параметрами модели
    """
    if available_memory_gb < 4:  # Очень ограниченная память
        return {
            "n_ctx": 1024,
            "width": 128,
            "layers": 4,
            "heads": 2,
            "text_embed_dim": 128,
            "batch_size": 1,
            "gradient_accumulation_steps": 8
        }
    elif available_memory_gb < 8:  # Ограниченная память (4-8 ГБ)
        return {
            "n_ctx": 2048,
            "width": 256,
            "layers": 6,
            "heads": 4,
            "text_embed_dim": 256,
            "batch_size": 1,
            "gradient_accumulation_steps": 4
        }
    elif available_memory_gb < 16:  # Средняя память (8-16 ГБ)
        return {
            "n_ctx": 4096,
            "width": 512,
            "layers": 8,
            "heads": 8,
            "text_embed_dim": 512,
            "batch_size": 2,
            "gradient_accumulation_steps": 2
        }
    else:  # Большая память (16+ ГБ)
        return {
            "n_ctx": 8192,
            "width": 768,
            "layers": 12,
            "heads": 12,
            "text_embed_dim": 768,
            "batch_size": 4,
            "gradient_accumulation_steps": 1
        }

def load_and_customize_config(args):
    """
    Загружает и настраивает конфигурацию модели на основе аргументов командной строки
    и доступных ресурсов
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        dict: Настроенная конфигурация модели
    """
    # Загружаем базовую конфигурацию
    config_path = args.config or f"configs/{args.model_type}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Файл конфигурации {config_path} не найден, создаём конфигурацию по умолчанию")
        config = {
            "n_ctx": 2048,
            "width": 256,
            "layers": 6,
            "heads": 4,
            "dropout": 0.1,
            "text_embed_dim": 256,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "input_channels": 3,
            "output_channels": 3
        }
    
    # Применяем автоматическое масштабирование на основе доступной памяти GPU, если включено
    if args.auto_scale and args.device != 'cpu' and torch.cuda.is_available():
        try:
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Автоматическое масштабирование для GPU с {available_memory_gb:.2f}GB памяти")
            
            # Получаем профиль масштабирования
            scaling_profile = get_memory_scaling_profile(available_memory_gb)
            
            # Применяем профиль к конфигурации
            for key, value in scaling_profile.items():
                if key in config:
                    config[key] = value
                    logger.info(f"Авто-масштабирование: {key} = {value}")
                elif hasattr(args, key):
                    setattr(args, key, value)
                    logger.info(f"Авто-масштабирование аргумента: {key} = {value}")
        except Exception as e:
            logger.warning(f"Ошибка при автоматическом масштабировании: {str(e)}")
    
    # Переопределяем параметры из командной строки (если указаны)
    override_params = [
        ("n_ctx", args.n_ctx),
        ("width", args.width),
        ("layers", args.layers),
        ("heads", args.heads),
        ("dropout", args.dropout),
        ("text_embed_dim", args.text_embed_dim)
    ]
    
    for param_name, param_value in override_params:
        if param_value is not None:
            original_value = config.get(param_name, "не задано")
            config[param_name] = param_value
            logger.info(f"Переопределяем {param_name}: {original_value} -> {param_value}")
    
    # Сохраняем настроенную конфигурацию для воспроизводимости
    customized_config_path = os.path.join(args.output_dir, f"{args.model_type}_customized.yaml")
    with open(customized_config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Настроенная конфигурация сохранена в {customized_config_path}")
    
    return config

def train_model(args):
    """Основная функция обучения модели"""
    # Устанавливаем seed для воспроизводимости
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Создаем директорию для моделей, если её нет
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Начало обучения модели {args.model_type} на устройстве {args.device}")
    
    device = torch.device(args.device)
    
    # Если включена подготовка данных, только её и делаем
    if args.prepare_data:
        from src.objaverse_dataset import prepare_objaverse_data
        prepare_objaverse_data(args.data_dir, args.num_objects if hasattr(args, 'num_objects') else 1000)
        return
    
    # Загружаем и настраиваем конфигурацию
    config = load_and_customize_config(args)
    
    # Используем настроенные параметры из конфига
    n_ctx = config.get("n_ctx", 2048)
    max_points = args.n_ctx if args.n_ctx is not None else n_ctx
    
    # Загружаем датасет
    try:
        # Проверка доступной GPU памяти
        if args.device != 'cpu' and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Доступно памяти GPU: {gpu_memory:.2f}GB")
            
            # Если параметры не определены автоматически или через CLI
            if not args.auto_scale and args.batch_size is None:
                # Если памяти мало, уменьшаем размер батча и увеличиваем gradient accumulation
                if gpu_memory < 8:  # Если менее 8 ГБ
                    logger.warning("Недостаточно памяти GPU, уменьшаем batch_size и увеличиваем gradient_accumulation_steps")
                    actual_batch_size = 1
                    gradient_accumulation_steps = args.batch_size
                    max_points = min(max_points, 2048)  # Ограничиваем количество точек
                else:
                    actual_batch_size = args.batch_size
                    gradient_accumulation_steps = 1
            else:
                # Используем параметры из авто-масштабирования или переданные вручную
                actual_batch_size = args.batch_size
                gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        else:
            actual_batch_size = args.batch_size
            gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
            
        # Загружаем данные с учетом оптимизированного batch_size
        from src.objaverse_dataset import get_objaverse_dataloader
        
        train_dataloader = get_objaverse_dataloader(
            args.data_dir, 
            batch_size=actual_batch_size, 
            split='train',
            num_workers=args.num_workers,
            max_points=max_points
        )
        
        val_dataloader = get_objaverse_dataloader(
            args.data_dir, 
            batch_size=actual_batch_size, 
            split='val',
            num_workers=args.num_workers,
            max_points=max_points
        )
        
        logger.info(f"Загружено {len(train_dataloader.dataset)} обучающих и {len(val_dataloader.dataset)} проверочных образцов")
        
        # Создаем диффузионную модель
        from src.models.diffusion import GaussianDiffusion
        
        diffusion = GaussianDiffusion.from_config(config)
        
        # Создаем или загружаем модель
        if args.skip_pretrained:
            logger.warning("Используется тестовая модель (без предварительного обучения)")
            
            if args.model_type == "text300M":
                from src.models.transformer import TextConditionedTransformer
                
                model = TextConditionedTransformer(
                    device=device,
                    input_channels=config.get("input_channels", 3),
                    output_channels=config.get("output_channels", 3),
                    n_ctx=config.get("n_ctx", 2048),
                    width=config.get("width", 256),
                    layers=config.get("layers", 6),
                    heads=config.get("heads", 4),
                    dropout=config.get("dropout", 0.1),
                    text_embed_dim=config.get("text_embed_dim", 256)
                )
            
            # Явно перемещаем модель на устройство
            model = model.to(device)
            
            # Проверяем, что все части модели находятся на правильном устройстве
            for param in model.parameters():
                if param.device != device and not (str(device) == 'cuda' and str(param.device) == 'cuda:0') and not (str(device) == 'cuda:0' and str(param.device) == 'cuda'):
                    logger.warning(f"Параметр модели находится на {param.device}, а не на {device}")
                    param.data = param.data.to(device)
                    
            # Включаем режим обучения
            model.train()
        else:
            # Загружаем предобученную модель
            from src.models.utils import load_model
            
            model = load_model(args.model_type, device=device)
            
            # Явно перемещаем модель на устройство
            model = model.to(device)
            model.train()
            
        # Создаем оптимизатор
        learning_rate = config.get("learning_rate", 1e-4)
        weight_decay = config.get("weight_decay", 0.01)
        
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Создаем scheduler для изменения learning rate
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=learning_rate * 0.1
        )
        
        # Используем новый API GradScaler
        scaler = GradScaler('cuda') if args.device != 'cpu' and torch.cuda.is_available() else GradScaler()
        
        # Выводим информацию о модели и её параметрах
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Модель {args.model_type} содержит {total_params:,} параметров, из них {trainable_params:,} обучаемых")
        logger.info(f"Параметры обучения: batch_size={actual_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
        logger.info(f"Параметры модели: n_ctx={config.get('n_ctx')}, width={config.get('width')}, layers={config.get('layers')}, heads={config.get('heads')}")
        
        # Обучение
        for epoch in range(args.num_epochs):
            logger.info(f"Эпоха {epoch+1}/{args.num_epochs}")
            
            model.train()
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Эпоха {epoch+1}")
            optimizer.zero_grad()  # Очищаем градиенты перед началом эпохи
            
            total_loss = 0.0
            
            for i, batch in enumerate(train_dataloader):
                try:
                    # Извлекаем данные
                    points = batch['points'].to(device)
                    text = batch.get('text', None)
                    
                    # Выбираем случайные моменты времени
                    t = torch.randint(
                        0, diffusion.num_timesteps, 
                        (points.shape[0],), 
                        device=device
                    ).long()
                    
                    # Готовим аргументы модели
                    model_kwargs = {}
                    if text is not None and hasattr(model, 'condition_on_text'):
                        model_kwargs['texts'] = text
                    
                    # Используем mixed precision
                    with autocast('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu'):
                        # Вычисляем потери
                        losses = diffusion.training_losses(model, points, t, model_kwargs=model_kwargs)
                        loss = losses["loss"] / gradient_accumulation_steps  # Нормализуем потерю
                    
                    # Обратное распространение с использованием scaler
                    scaler.scale(loss).backward()
                    
                    # Накапливаем потери для вывода
                    total_loss += loss.item() * gradient_accumulation_steps
                    
                    # Обновляем веса каждые gradient_accumulation_steps шагов
                    if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps, lr=optimizer.param_groups[0]['lr'])
                    
                except Exception as e:
                    logger.error(f"Ошибка в шаге обучения: {str(e)}")
                    # Продолжаем обучение даже если возникла ошибка
                    optimizer.zero_grad()  # Сбрасываем градиенты
                    
            progress_bar.close()
            
            # Сохраняем модель после каждой эпохи
            from src.models.utils import save_model
            
            save_path = save_model(model, args.model_type, args.output_dir)
            logger.info(f"Завершена эпоха {epoch+1}, модель сохранена")
            
            # Средняя потеря за эпоху
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Средняя потеря за эпоху: {avg_loss:.4f}")
            
            # Освобождаем память после каждой эпохи
            empty_cache_and_collect_garbage()
            
            # Запускаем валидацию
            if (epoch + 1) % args.val_freq == 0:
                val_loss = validate_model(model, val_dataloader, device, diffusion)
                logger.info(f"Валидация эпохи {epoch+1}: потеря = {val_loss['loss']:.4f}")
            
            # Обновляем scheduler
            scheduler.step()
            
        logger.info("Обучение завершено!")
            
    except Exception as e:
        logger.error(f"Ошибка при обучении: {str(e)}")
        raise e
    finally:
        # Освобождаем память перед выходом
        if args.device != 'cpu' and torch.cuda.is_available():
            empty_cache_and_collect_garbage()

if __name__ == "__main__":
    args = parse_args()
    train_model(args) 