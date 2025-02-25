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

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.modelnet_dataset import ModelNetDataset
from models.text_to_3d_model import TextTo3DModel
from utils.voxelization import process_voxel_prediction, voxel_to_mesh, save_mesh

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
    
    return config

def create_dataloaders(config):
    """
    Создает загрузчики данных для обучения, валидации и тестирования.
    
    Args:
        config: Конфигурация.
        
    Returns:
        tuple: Кортеж из трех загрузчиков данных (train, val, test).
    """
    # Проверка наличия директории с данными
    if not os.path.exists(config.data.dataset_path):
        os.makedirs(config.data.dataset_path, exist_ok=True)
        print(f"Директория {config.data.dataset_path} создана. Пожалуйста, загрузите датасет ModelNet40.")
        print("Вы можете скачать его с http://modelnet.cs.princeton.edu/")
        exit(1)
    
    # Создание датасетов
    full_dataset = ModelNetDataset(
        root_dir=config.data.dataset_path,
        split='train',
        voxel_resolution=config.data.voxel_resolution,
        text_augmentation=config.data.augmentation
    )
    
    test_dataset = ModelNetDataset(
        root_dir=config.data.dataset_path,
        split='test',
        voxel_resolution=config.data.voxel_resolution,
        text_augmentation=False
    )
    
    # Разделение на обучающую и валидационную выборки
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * config.data.train_split)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")
    print(f"Размер тестовой выборки: {len(test_dataset)}")
    
    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def compute_loss(pred_voxels, target_voxels):
    """
    Вычисляет функцию потерь между предсказанными и целевыми воксельными сетками.
    
    Args:
        pred_voxels (torch.Tensor): Предсказанные воксельные сетки.
        target_voxels (torch.Tensor): Целевые воксельные сетки.
        
    Returns:
        torch.Tensor: Значение функции потерь.
    """
    # Бинарная кросс-энтропия для воксельных сеток
    bce_loss = nn.BCELoss()(pred_voxels, target_voxels)
    
    # Дополнительная L1 потеря для улучшения деталей
    l1_loss = nn.L1Loss()(pred_voxels, target_voxels)
    
    # Комбинированная потеря
    total_loss = bce_loss + 0.1 * l1_loss
    
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
    
    for batch in pbar:
        # Получение данных из батча
        voxels = batch['voxels'].to(device)
        text_prompts = batch['text']
        
        # Обнуление градиентов
        optimizer.zero_grad()
        
        # Прямой проход
        pred_voxels = model(text_prompts)
        
        # Вычисление потерь
        loss, bce_loss, l1_loss = compute_loss(pred_voxels, voxels)
        
        # Обратное распространение ошибки
        loss.backward()
        
        # Ограничение градиентов (если нужно)
        if config.training.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad_norm)
        
        # Шаг оптимизатора
        optimizer.step()
        
        # Вычисление IoU
        iou = compute_iou(pred_voxels.detach(), voxels)
        
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
    
    # Вычисление средних значений метрик
    avg_loss = total_loss / len(dataloader)
    avg_bce_loss = total_bce_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return {
        'loss': avg_loss,
        'bce_loss': avg_bce_loss,
        'l1_loss': avg_l1_loss,
        'iou': avg_iou
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
            
            # Прямой проход
            pred_voxels = model(text_prompts)
            
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
                'val_loss': loss.item(),
                'val_bce': bce_loss.item(),
                'val_l1': l1_loss.item(),
                'val_iou': iou
            })
    
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
    Основная функция обучения модели.
    
    Args:
        config: Конфигурация.
    """
    # Установка seed для воспроизводимости
    set_seed(config.seed)
    
    # Определение устройства для вычислений
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Создание загрузчиков данных
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Инициализация модели
    model = TextTo3DModel(config)
    model.to(device)
    
    # Инициализация оптимизатора
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Инициализация планировщика скорости обучения
    if config.training.lr_scheduler.type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs - config.training.lr_scheduler.warmup_epochs
        )
    else:
        scheduler = None
    
    # Инициализация логгеров
    if config.logging.wandb.use:
        wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            config=vars(config)
        )
        wandb.watch(model)
    
    if config.logging.tensorboard.use:
        tb_log_dir = os.path.join(config.log_dir, config.experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Переменные для отслеживания лучшей модели
    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience_counter = 0
    
    # Основной цикл обучения
    for epoch in range(config.training.num_epochs):
        # Обучение на одной эпохе
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)
        
        # Валидация модели
        val_metrics = validate(model, val_loader, device, epoch, config)
        
        # Обновление планировщика скорости обучения
        if scheduler is not None and epoch >= config.training.lr_scheduler.warmup_epochs:
            scheduler.step()
        
        # Логирование метрик
        if config.logging.wandb.use:
            wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})
        
        if config.logging.tensorboard.use:
            for key, value in {**train_metrics, **val_metrics}.items():
                tb_writer.add_scalar(key, value, epoch)
        
        # Вывод метрик
        print(f"Эпоха {epoch+1}/{config.training.num_epochs}")
        print(f"Обучение: loss={train_metrics['loss']:.4f}, iou={train_metrics['iou']:.4f}")
        print(f"Валидация: loss={val_metrics['val_loss']:.4f}, iou={val_metrics['val_iou']:.4f}")
        
        # Проверка на лучшую модель по IoU
        is_best = val_metrics['val_iou'] > best_val_iou
        if is_best:
            best_val_iou = val_metrics['val_iou']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Сохранение чекпоинта
        if (epoch + 1) % config.training.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, {**train_metrics, **val_metrics}, config, is_best)
        
        # Визуализация примеров
        if (epoch + 1) % config.training.evaluate_interval == 0:
            visualize_samples(model, val_loader, device, epoch, config)
        
        # Ранняя остановка
        if patience_counter >= config.training.early_stopping.patience:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break
    
    # Закрытие логгеров
    if config.logging.tensorboard.use:
        tb_writer.close()
    
    if config.logging.wandb.use:
        wandb.finish()
    
    # Тестирование лучшей модели
    print("Тестирование лучшей модели...")
    # Загрузка лучшей модели
    best_checkpoint = torch.load(os.path.join(config.checkpoint_dir, "best_model.pt"))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Тестирование
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