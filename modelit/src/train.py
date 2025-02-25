import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
import time
import datetime
import sys
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.modelnet_dataset import ModelNetDataset
from models.text_to_3d_model import TextTo3DModel
from utils.voxelization import process_voxel_prediction, voxel_to_mesh, save_mesh

# Оптимизация производительности CUDA за счёт детерминизма
cudnn.benchmark = False
cudnn.deterministic = True

def set_seed(seed):
    """
    Устанавливает seed для воспроизводимости результатов.
    
    Args:
        seed (int): Значение seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    
    # Убедимся, что значение voxel_dim в модели соответствует voxel_resolution в данных
    if hasattr(config, 'data') and hasattr(config, 'model') and hasattr(config.model, 'shape_generator'):
        if hasattr(config.data, 'voxel_resolution') and hasattr(config.model.shape_generator, 'voxel_dim'):
            # Синхронизируем размерности воксельной сетки
            voxel_res = config.data.voxel_resolution
            config.model.shape_generator.voxel_dim = voxel_res
            print(f"Синхронизированы размерности вокселей: data.voxel_resolution = {voxel_res}, "
                  f"model.shape_generator.voxel_dim = {config.model.shape_generator.voxel_dim}")
    
    return config

def create_dataloaders(config):
    """
    Создает загрузчики данных для обучения, валидации и тестирования.
    
    Args:
        config: Конфигурация.
        
    Returns:
        tuple: Кортеж (train_loader, val_loader, test_loader).
    """
    # Создание обучающего датасета
    train_dataset = ModelNetDataset(
        root_dir=config.data.dataset_path,
        split='train',
        text_augmentation=config.data.augmentation,
        voxel_resolution=config.data.voxel_resolution
    )
    
    # Определение размеров выборок
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * config.data.train_split)
    val_size = int(dataset_size * config.data.validation_split)
    test_size = dataset_size - train_size - val_size
    
    # Разделение на обучающую, валидационную и тестовую выборки
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size]
    )
    
    # Дополнительный тестовый датасет (необработанный)
    test_dataset_raw = ModelNetDataset(
        root_dir=config.data.dataset_path,
        split='test',
        text_augmentation=False,
        voxel_resolution=config.data.voxel_resolution
    )
    
    # Определение, нужно ли использовать pin_memory
    pin_memory = False
    if hasattr(config.data, 'pin_memory'):
        pin_memory = config.data.pin_memory
    
    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset_raw,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader

def compute_loss(pred_voxels, target_voxels):
    """
    Вычисление функции потерь.
    
    Args:
        pred_voxels: Предсказанные воксели.
        target_voxels: Целевые воксели.
        
    Returns:
        tuple: Кортеж (total_loss, bce_loss, l1_loss).
    """
    # Binary Cross Entropy loss
    bce_loss = nn.BCEWithLogitsLoss()(pred_voxels, target_voxels)
    
    # L1 loss для регуляризации
    l1_loss = nn.L1Loss()(torch.sigmoid(pred_voxels), target_voxels)
    
    # Общая потеря - взвешенная сумма BCE и L1
    bce_weight = 1.0
    l1_weight = 0.1
    
    # Проверяем, есть ли в конфигурации веса для потерь
    config = SimpleNamespace()
    if hasattr(config, 'loss') and hasattr(config.loss, 'bce_weight'):
        bce_weight = config.loss.bce_weight
    if hasattr(config, 'loss') and hasattr(config.loss, 'l1_weight'):
        l1_weight = config.loss.l1_weight
        
    total_loss = bce_weight * bce_loss + l1_weight * l1_loss
    
    return total_loss, bce_loss, l1_loss

def compute_iou(pred_voxels, target_voxels, threshold=0.5):
    """
    Вычисляет IoU (Intersection over Union) между предсказанными и целевыми воксельными сетками.
    
    Args:
        pred_voxels (torch.Tensor): Предсказанные воксельные сетки.
        target_voxels (torch.Tensor): Целевые воксельные сетки.
        threshold (float): Порог бинаризации.
        
    Returns:
        float: Значение IoU.
    """
    # Бинаризация предсказаний
    pred_binary = (pred_voxels > threshold).float()
    target_binary = (target_voxels > threshold).float()
    
    # Вычисление пересечения и объединения
    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection
    
    # Вычисление IoU
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.item()

def train_epoch(model, dataloader, optimizer, device, epoch, config):
    """
    Обучение модели на одной эпохе.
    
    Args:
        model: Модель для обучения.
        dataloader: Загрузчик данных.
        optimizer: Оптимизатор.
        device: Устройство для вычислений.
        epoch: Номер текущей эпохи.
        config: Конфигурация.
        
    Returns:
        dict: Словарь с метриками обучения.
    """
    model.train()
    total_loss = 0.0
    total_bce_loss = 0.0
    total_l1_loss = 0.0
    total_iou = 0.0
    
    # Прогресс-бар
    pbar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{config.training.num_epochs} [Обучение]")
    
    for i, batch in enumerate(pbar):
        # Получение данных из батча
        voxels = batch['voxels'].to(device)
        text_prompts = batch['text']
        
        # Обнуление градиентов
        optimizer.zero_grad()
        
        # Прямой проход с использованием смешанной точности
        with torch.cuda.amp.autocast(enabled=config.training.use_mixed_precision):
            pred_voxels = model(text_prompts)
            
            # Проверка размерностей тензоров
            if pred_voxels.shape != voxels.shape:
                print(f"Предупреждение: Размеры тензоров не совпадают. "
                      f"Предсказания: {pred_voxels.shape}, Целевые: {voxels.shape}")
                
                # Изменяем размер выходного тензора модели, чтобы он соответствовал целевому
                target_size = voxels.shape[2]  # Получаем целевой размер (D, H, W)
                pred_size = pred_voxels.shape[2]  # Получаем текущий размер предсказаний
                
                if target_size != pred_size:
                    # Интерполируем тензор до нужного размера
                    pred_voxels = torch.nn.functional.interpolate(
                        pred_voxels, 
                        size=(target_size, target_size, target_size),
                        mode='trilinear',
                        align_corners=False
                    )
            
            # Вычисление потерь
            loss, bce_loss, l1_loss = compute_loss(pred_voxels, voxels)
        
        # Нормализация потерь из-за аккумуляции градиентов
        loss = loss / config.training.accumulation_steps
        
        # Обратное распространение ошибки
        loss.backward()
        
        # Ограничение градиентов (если нужно)
        if config.training.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad_norm)
        
        # Обновляем веса только после накопления нескольких градиентов
        if (i + 1) % config.training.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Очистка кэша CUDA для экономии памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Вычисление IoU
        iou = compute_iou(pred_voxels.detach(), voxels)
        
        # Обновление метрик
        total_loss += loss.item() * config.training.accumulation_steps  # Учитываем масштабирование
        total_bce_loss += bce_loss.item()
        total_l1_loss += l1_loss.item()
        total_iou += iou
        
        # Обновление прогресс-бара
        pbar.set_postfix({
            'loss': loss.item() * config.training.accumulation_steps,
            'bce': bce_loss.item(),
            'l1': l1_loss.item(),
            'iou': iou
        })
        
        # Освобождаем память от ненужных переменных
        del voxels, pred_voxels, loss, bce_loss, l1_loss
    
    # Выполнение последнего шага оптимизатора, если остались накопленные градиенты
    if len(dataloader) % config.training.accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Вычисление средних значений метрик
    avg_loss = total_loss / len(dataloader)
    avg_bce_loss = total_bce_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return {
        'train_loss': avg_loss,
        'train_bce_loss': avg_bce_loss,
        'train_l1_loss': avg_l1_loss,
        'train_iou': avg_iou
    }

def validate(model, dataloader, device, epoch, config):
    """
    Валидация модели.
    
    Args:
        model: Модель для валидации.
        dataloader: Загрузчик данных.
        device: Устройство для вычислений.
        epoch: Номер текущей эпохи.
        config: Конфигурация.
        
    Returns:
        dict: Словарь с метриками валидации.
    """
    model.eval()
    total_loss = 0.0
    total_bce_loss = 0.0
    total_l1_loss = 0.0
    total_iou = 0.0
    
    # Прогресс-бар
    pbar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{config.training.num_epochs} [Валидация]")
    
    with torch.no_grad():
        for batch in pbar:
            # Получение данных из батча
            voxels = batch['voxels'].to(device)
            text_prompts = batch['text']
            
            # Прямой проход с использованием смешанной точности
            with torch.cuda.amp.autocast(enabled=config.training.use_mixed_precision):
                pred_voxels = model(text_prompts)
                
                # Проверка размерностей тензоров
                if pred_voxels.shape != voxels.shape:
                    print(f"Предупреждение: Размеры тензоров не совпадают. "
                          f"Предсказания: {pred_voxels.shape}, Целевые: {voxels.shape}")
                    
                    # Изменяем размер выходного тензора модели, чтобы он соответствовал целевому
                    target_size = voxels.shape[2]  # Получаем целевой размер (D, H, W)
                    pred_size = pred_voxels.shape[2]  # Получаем текущий размер предсказаний
                    
                    if target_size != pred_size:
                        # Интерполируем тензор до нужного размера
                        pred_voxels = torch.nn.functional.interpolate(
                            pred_voxels, 
                            size=(target_size, target_size, target_size),
                            mode='trilinear',
                            align_corners=False
                        )
                
                # Вычисление потерь
                loss, bce_loss, l1_loss = compute_loss(pred_voxels, voxels)
            
            # Вычисление IoU
            iou = compute_iou(pred_voxels, voxels)
            
            # Обновление метрик
            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_l1_loss += l1_loss.item()
            total_iou += iou
            
            # Обновление прогресс-бара
            pbar.set_postfix({
                'loss': loss.item(),
                'bce': bce_loss.item(),
                'l1': l1_loss.item(),
                'iou': iou
            })
            
            # Освобождаем память
            del voxels, pred_voxels, loss, bce_loss, l1_loss
            
            # Очистка кэша CUDA для экономии памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Вычисление средних значений метрик
    avg_loss = total_loss / len(dataloader)
    avg_bce_loss = total_bce_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return {
        'val_loss': avg_loss,
        'val_bce_loss': avg_bce_loss,
        'val_l1_loss': avg_l1_loss,
        'val_iou': avg_iou
    }

def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """
    Сохраняет чекпоинт модели.
    
    Args:
        model: Модель для сохранения.
        optimizer: Оптимизатор.
        epoch: Номер текущей эпохи.
        metrics: Метрики модели.
        config: Конфигурация.
        is_best (bool): Является ли модель лучшей на данный момент.
    """
    # Создание директории для чекпоинтов, если она не существует
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Путь для сохранения чекпоинта
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    # Сохранение чекпоинта
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    
    print(f"Чекпоинт сохранен: {checkpoint_path}")
    
    # Если это лучшая модель, сохраняем ее отдельно
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, best_path)
        print(f"Лучшая модель сохранена: {best_path}")

def visualize_samples(model, dataloader, device, epoch, config):
    """
    Визуализирует примеры генерации 3D объектов.
    
    Args:
        model: Модель для генерации.
        dataloader: Загрузчик данных.
        device: Устройство для вычислений.
        epoch: Номер текущей эпохи.
        config: Конфигурация.
    """
    model.eval()
    
    # Создание директории для визуализаций
    vis_dir = os.path.join(config.evaluation.visualization.output_dir, f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Получение нескольких примеров из датасета
    batch = next(iter(dataloader))
    voxels = batch['voxels'].to(device)
    text_prompts = batch['text']
    categories = batch['category']
    
    # Ограничение количества примеров
    num_samples = min(config.evaluation.visualization.num_samples, len(voxels))
    
    with torch.no_grad():
        # Генерация 3D объектов
        pred_voxels = model(text_prompts[:num_samples])
        
        # Проверка размерностей тензоров
        if pred_voxels.shape[2:] != voxels.shape[2:]:
            print(f"Предупреждение: Размеры тензоров не совпадают при визуализации. "
                  f"Предсказания: {pred_voxels.shape}, Целевые: {voxels.shape}")
            
            # Изменяем размер выходного тензора модели, чтобы он соответствовал целевому
            target_size = voxels.shape[2]  # Получаем целевой размер (D, H, W)
            
            # Интерполируем тензор до нужного размера
            pred_voxels = torch.nn.functional.interpolate(
                pred_voxels, 
                size=(target_size, target_size, target_size),
                mode='trilinear',
                align_corners=False
            )
        
        # Сохранение результатов
        for i in range(num_samples):
            # Получение предсказанного и целевого воксельного представления
            pred_voxel = pred_voxels[i, 0].cpu().numpy()
            target_voxel = voxels[i, 0].cpu().numpy()
            
            # Преобразование в меш и сохранение
            pred_mesh = process_voxel_prediction(pred_voxel)
            target_mesh = voxel_to_mesh(target_voxel)
            
            if pred_mesh is not None:
                # Сохранение предсказанного меша
                pred_path = os.path.join(vis_dir, f"pred_{i}_{categories[i]}.obj")
                save_mesh(pred_mesh, pred_path)
                
                # Сохранение целевого меша
                target_path = os.path.join(vis_dir, f"target_{i}_{categories[i]}.obj")
                save_mesh(target_mesh, target_path)
                
                # Сохранение текстового промпта
                prompt_path = os.path.join(vis_dir, f"prompt_{i}_{categories[i]}.txt")
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(text_prompts[i])

def train(config):
    """
    Полный процесс обучения модели.
    
    Args:
        config: Конфигурация.
        
    Returns:
        TextTo3DModel: Обученная модель.
    """
    # Установка генератора случайных чисел
    set_seed(config.seed)
    
    # Определение устройства для вычислений
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Отслеживание использования памяти
    if hasattr(config, 'logging') and hasattr(config.logging, 'memory_tracking') and config.logging.memory_tracking and torch.cuda.is_available():
        print(f"Начальное использование CUDA памяти: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Загрузка датасета и создание загрузчиков данных
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Создание директорий для логов и чекпоинтов
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Инициализация логгеров
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Инициализация модели
    model = TextTo3DModel(config)
    model = model.to(device)
    
    # Включение gradient checkpointing, если указано
    if hasattr(config, 'optimization') and hasattr(config.optimization, 'gradient_checkpointing') and config.optimization.gradient_checkpointing:
        print("Включен gradient checkpointing для экономии памяти")
        # Включаем gradient checkpointing для transformer-блоков
        model.enable_gradient_checkpointing()
    
    # Оптимизатор
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Планировщик скорости обучения
    if config.training.lr_scheduler.type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs - config.training.lr_scheduler.warmup_epochs
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Инициализация переменных для ранней остановки
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Основной цикл обучения
    print(f"Начало обучения на {config.training.num_epochs} эпохах")
    print(f"Размер батча: {config.data.batch_size}, аккумуляция градиентов: {config.training.accumulation_steps}")
    print(f"Эффективный размер батча: {config.data.batch_size * config.training.accumulation_steps}")
    
    # Проверка доступности функций отслеживания памяти CUDA
    has_memory_tracking = (hasattr(config, 'logging') and 
                          hasattr(config.logging, 'memory_tracking') and 
                          config.logging.memory_tracking and 
                          torch.cuda.is_available())
    
    has_reset_peak_memory = hasattr(torch.cuda, 'reset_peak_memory_stats')
    has_max_memory_allocated = hasattr(torch.cuda, 'max_memory_allocated')
    
    try:
        for epoch in range(config.training.num_epochs):
            # Измерение памяти перед эпохой
            if has_memory_tracking:
                if has_reset_peak_memory:
                    torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
                print(f"Память CUDA перед эпохой {epoch+1}: {start_memory / 1e9:.2f} GB")
            
            # Обучение на одной эпохе
            train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)
            
            # Измерение пиковой памяти
            if has_memory_tracking and has_max_memory_allocated:
                peak_memory = torch.cuda.max_memory_allocated()
                print(f"Пиковое использование CUDA памяти в эпохе {epoch+1}: {peak_memory / 1e9:.2f} GB")
            
            # Логирование метрик обучения
            writer.add_scalar('Train/Loss', train_metrics['train_loss'], epoch)
            writer.add_scalar('Train/BCE_Loss', train_metrics['train_bce_loss'], epoch)
            writer.add_scalar('Train/L1_Loss', train_metrics['train_l1_loss'], epoch)
            writer.add_scalar('Train/IoU', train_metrics['train_iou'], epoch)
            
            # Валидация модели
            if (epoch + 1) % config.training.evaluate_interval == 0:
                val_metrics = validate(model, val_loader, device, epoch, config)
                
                # Логирование метрик валидации
                writer.add_scalar('Val/Loss', val_metrics['val_loss'], epoch)
                writer.add_scalar('Val/BCE_Loss', val_metrics['val_bce_loss'], epoch)
                writer.add_scalar('Val/L1_Loss', val_metrics['val_l1_loss'], epoch)
                writer.add_scalar('Val/IoU', val_metrics['val_iou'], epoch)
                
                # Визуализация примеров
                if (epoch + 1) % config.training.save_interval == 0:
                    visualize_samples(model, val_loader, device, epoch, config)
                
                # Проверка на улучшение
                if val_metrics['val_loss'] < best_val_loss - config.training.early_stopping.min_delta:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    
                    # Сохранение лучшей модели
                    save_checkpoint(model, optimizer, epoch, val_metrics, config, is_best=True)
                else:
                    patience_counter += 1
                
                # Проверка на раннюю остановку
                if patience_counter >= config.training.early_stopping.patience:
                    print(f"Раннее прекращение обучения на эпохе {epoch+1}")
                    break
            
            # Обновление планировщика скорости обучения
            scheduler.step()
            
            # Сохранение чекпоинта
            if (epoch + 1) % config.training.save_interval == 0:
                save_checkpoint(model, optimizer, epoch, train_metrics, config)
            
            # Очистка памяти CUDA после каждой эпохи
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if has_memory_tracking:
                    end_memory = torch.cuda.memory_allocated()
                    print(f"Память CUDA после эпохи {epoch+1}: {end_memory / 1e9:.2f} GB")
    
    except KeyboardInterrupt:
        print("Обучение прервано пользователем")
    
    # Тестирование финальной модели
    print("Проведение финального тестирования...")
    test_metrics = validate(model, test_loader, device, -1, config)
    print(f"Тестирование: loss={test_metrics['val_loss']:.4f}, iou={test_metrics['val_iou']:.4f}")
    
    # Визуализация тестовых примеров
    visualize_samples(model, test_loader, device, -1, config)
    
    return model

def main():
    """
    Основная функция.
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Обучение модели Text-to-3D")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Создание директорий для логов и чекпоинтов
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.evaluation.visualization.output_dir, exist_ok=True)
    
    # Запуск обучения
    train(config)

if __name__ == "__main__":
    main() 