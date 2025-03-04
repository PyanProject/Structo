#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from types import SimpleNamespace
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter
import random

# Добавляем путь к пакету src в sys.path
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.append(current_path)

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.text_to_3d_model import TextTo3DModel
from src.data.modelnet_dataset import ModelNetDataset
from src.utils.metrics import compute_iou, compute_chamfer_distance, compute_f_score
from src.utils.visualization import visualize_voxel_grid, create_3d_visualization
import src.utils.training_utils as training_utils

# Настройка логирования
def setup_logging(log_dir, experiment_name):
    """
    Sets up logging for the training process.
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Format log filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{experiment_name}_{current_time}.log")
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Fix stdout encoding for Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    
    logger = logging.getLogger()
    logger.info(f"Logging configured. Log file: {log_filename}")
    
    return logger

def load_config(config_path):
    """
    Загружает конфигурацию из YAML файла.
    """
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

def setup_training_environment(config):
    """
    Sets up the training environment based on the provided configuration.
    """
    # Установка случайного зерна для воспроизводимости, если указано в конфигурации
    if hasattr(config, 'seed') and config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        logging.info(f"Random seed set: {config.seed}")
    
    # Определение устройства для обучения
    if hasattr(config, 'device') and config.device.lower() == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA is not available, although specified in config. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and getattr(config, 'device', '').lower() == 'cuda' else 'cpu')
    
    logging.info(f"Using device: {device}")
    
    return device

def init_logging_platforms(config, experiment_name):
    """
    Initializes logging platforms such as TensorBoard and W&B.
    """
    # TensorBoard
    tensorboard_log_dir = os.path.join(config.log_dir, 'tensorboard', experiment_name)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logging.info(f"TensorBoard logging configured. Directory: {tensorboard_log_dir}")
    
    # Weights & Biases - if enabled in config
    if hasattr(config, 'use_wandb') and config.use_wandb:
        try:
            import wandb
            wandb.init(project=getattr(config, 'wandb_project', 'text_to_3d'),
                       name=experiment_name,
                       config=config.__dict__)
            logging.info("Weights & Biases logging configured.")
        except ImportError:
            logging.warning("Weights & Biases requested but not installed. Skipping W&B initialization.")
        except Exception as e:
            logging.warning(f"Failed to initialize Weights & Biases: {str(e)}")
    
    return tensorboard_writer

def get_transforms(config, split):
    """
    Creates data transformations for the specified split.
    
    Args:
        config: Configuration object
        split: Dataset split ('train' or 'val')
        
    Returns:
        callable: Transform function to apply to the data
    """
    # In this simple implementation, we return None as no transforms are needed
    # This can be expanded later to include data augmentation for training
    return None

def create_dataloaders(config):
    """
    Creates data loaders for training and validation sets.
    """
    dataset_name = config.data.dataset_name
    dataset_path = config.data.dataset_path
    
    logging.info(f"Loading {dataset_name} dataset from {dataset_path}")
    
    # Создание и настройка трансформации данных в зависимости от типа датасета
    train_dataset = ModelNetDataset(
        root_dir=dataset_path,
        split='train',
        transform=get_transforms(config, 'train'),
        text_augmentation=config.data.augmentation
    )
    
    val_dataset = ModelNetDataset(
        root_dir=dataset_path,
        split='test',
        transform=get_transforms(config, 'val'),
        text_augmentation=False
    )
    
    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=getattr(config.data, 'pin_memory', False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=getattr(config.data, 'pin_memory', False)
    )
    
    logging.info(f"Data loaders created. Training samples: {len(train_dataset)}, "
                 f"validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def create_model(config):
    """
    Создание модели на основе конфигурации.
    """
    model = TextTo3DModel(config)
    
    # Подсчет количества параметров
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Создана модель с {num_params:,} обучаемыми параметрами")
    
    return model

def create_loss_function(config):
    """
    Создание функции потерь на основе конфигурации.
    """
    # Веса компонентов функции потерь
    bce_weight = getattr(config.loss, 'bce_weight', 1.0)
    l1_weight = getattr(config.loss, 'l1_weight', 0.0)
    perceptual_weight = getattr(config.loss, 'perceptual_weight', 0.0)
    chamfer_weight = getattr(config.loss, 'chamfer_weight', 0.0)
    
    # Базовые функции потерь
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Создание композитной функции потерь
    def composite_loss(predictions, targets):
        # Основная потеря - BCE
        loss = bce_weight * bce_loss(predictions, targets)
        
        # Потеря L1 (регуляризация)
        if l1_weight > 0:
            l1 = nn.L1Loss()(torch.sigmoid(predictions), targets)
            loss += l1_weight * l1
        
        # Добавление других компонентов по мере необходимости
        if perceptual_weight > 0:
            # Реализация перцептивной потери (в будущих обновлениях)
            pass
        
        if chamfer_weight > 0:
            # Реализация потери Чемфера (в будущих обновлениях)
            pass
        
        return loss
    
    return composite_loss

def create_optimizer(model, config):
    """
    Создание оптимизатора на основе конфигурации.
    """
    optimizer_type = getattr(config.training, 'optimizer', 'adam')
    learning_rate = config.training.learning_rate
    weight_decay = getattr(config.training, 'weight_decay', 0.0)
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Неподдерживаемый тип оптимизатора: {optimizer_type}")
    
    return optimizer

def create_lr_scheduler(optimizer, config):
    """
    Creates learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer.
        config: Configuration.
        
    Returns:
        Learning rate scheduler.
    """
    if not hasattr(config.training, 'lr_scheduler'):
        return None
    
    lr_scheduler_config = config.training.lr_scheduler
    lr_scheduler_type = getattr(lr_scheduler_config, 'type', 'constant')
    
    if lr_scheduler_type == 'constant':
        return None
    
    elif lr_scheduler_type == 'step':
        step_size = getattr(lr_scheduler_config, 'step_size', 30)
        gamma = getattr(lr_scheduler_config, 'gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif lr_scheduler_type == 'multistep':
        milestones = getattr(lr_scheduler_config, 'milestones', [30, 60, 90])
        gamma = getattr(lr_scheduler_config, 'gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif lr_scheduler_type == 'exponential':
        gamma = getattr(lr_scheduler_config, 'gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif lr_scheduler_type == 'cosine':
        num_epochs = config.training.num_epochs
        warmup_epochs = getattr(lr_scheduler_config, 'warmup_epochs', 0)
        base_lr = config.training.learning_rate
        
        if warmup_epochs > 0:
            # Создаем составной планировщик с разогревом
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs - warmup_epochs,
                eta_min=base_lr * 0.001
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            # Простой косинусный планировщик без разогрева
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs,
                eta_min=base_lr * 0.001
            )
    
    elif lr_scheduler_type == 'plateau':
        patience = getattr(lr_scheduler_config, 'patience', 10)
        factor = getattr(lr_scheduler_config, 'factor', 0.5)
        threshold = getattr(lr_scheduler_config, 'threshold', 0.01)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor,
            patience=patience, 
            threshold=threshold
        )
    
    else:
        logging.warning(f"Unknown scheduler type: {lr_scheduler_type}. Using default scheduler.")
        return None

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config):
    """
    Training the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    
    # Training parameters
    use_mixed_precision = getattr(config.training, 'use_mixed_precision', False)
    accumulation_steps = getattr(config.training, 'accumulation_steps', 1)
    clip_grad_norm = getattr(config.training, 'clip_grad_norm', None)
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Training]", leave=False)
    
    # Metrics counters
    total_iou = 0.0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(pbar):
        # Get voxels and text data from batch
        voxels = batch_data['voxels']
        texts = batch_data['text']
        
        # Move data to device
        voxels = voxels.to(device)
        
        # Zero gradients at first accumulation step
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(texts)
                loss = loss_fn(outputs, voxels)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
            # Backward propagation with scaler
            scaler.scale(loss).backward()
            
            # Update weights at last accumulation step
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                scaler.step(optimizer)
                scaler.update()
        else:
            # Standard training mode
            outputs = model(texts)
            loss = loss_fn(outputs, voxels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward propagation
            loss.backward()
            
            # Update weights at last accumulation step
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Apply sigmoid to outputs to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Binarize with threshold 0.5
            binary_output = (probs > 0.5).float()
            
            # Calculate IoU
            iou = compute_iou(binary_output, voxels)
            total_iou += iou * voxels.size(0)
            total_samples += voxels.size(0)
        
        # Update running_loss and progress bar
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'iou': total_iou / total_samples
        })
    
    # Calculate average metrics
    avg_loss = running_loss / len(train_loader)
    avg_iou = total_iou / total_samples
    
    logging.info(f"Epoch {epoch+1} [Training] - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")
    
    return avg_loss, avg_iou

def validate(model, val_loader, loss_fn, device, epoch, config):
    """
    Validation of the model.
    """
    model.eval()
    running_loss = 0.0
    
    # Metrics counters
    total_iou = 0.0
    total_f_score = 0.0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Validation]", leave=False)
    
    # Select examples for visualization
    visualize_idx = np.random.randint(0, len(val_loader))
    visualization_data = None
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar):
            # Get voxels and text data from batch
            voxels = batch_data['voxels']
            texts = batch_data['text']
            
            # Move data to device
            voxels = voxels.to(device)
            
            # Forward pass
            outputs = model(texts)
            loss = loss_fn(outputs, voxels)
            
            # Calculate metrics
            probs = torch.sigmoid(outputs)
            binary_output = (probs > 0.5).float()
            
            # IoU
            iou = compute_iou(binary_output, voxels)
            total_iou += iou * voxels.size(0)
            
            # F-score
            f_score = compute_f_score(binary_output, voxels)
            total_f_score += f_score * voxels.size(0)
            
            total_samples += voxels.size(0)
            
            # Update running_loss and progress bar
            running_loss += loss.item()
            pbar.set_postfix({
                'val_loss': running_loss / (batch_idx + 1),
                'val_iou': total_iou / total_samples
            })
            
            # Select example for visualization
            if batch_idx == visualize_idx:
                visualization_data = {
                    'inputs': voxels.detach().cpu().numpy(),
                    'outputs': probs.detach().cpu().numpy(),
                    'texts': texts,
                }
    
    # Calculate average metrics
    avg_loss = running_loss / len(val_loader)
    avg_iou = total_iou / total_samples
    avg_f_score = total_f_score / total_samples
    
    logging.info(f"Epoch {epoch+1} [Validation] - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, F-score: {avg_f_score:.4f}")
    
    return avg_loss, avg_iou, avg_f_score, visualization_data

def save_visualizations(visualization_data, epoch, output_dir):
    """
    Сохранение визуализаций предсказаний модели.
    """
    if visualization_data is None:
        return
    
    # Создание директории для визуализаций
    os.makedirs(output_dir, exist_ok=True)
    
    # Получение данных
    inputs = visualization_data['inputs']
    outputs = visualization_data['outputs']
    texts = visualization_data['texts']
    
    # Выбор нескольких примеров для визуализации
    num_examples = min(3, inputs.shape[0])
    
    for i in range(num_examples):
        # Получение данных для визуализации
        input_voxel = inputs[i, 0]  # [D, H, W]
        output_voxel = outputs[i, 0]  # [D, H, W]
        text = texts[i]
        
        # Визуализация входных данных
        input_img_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{i+1}_input.png")
        visualize_voxel_grid(input_voxel, save_path=input_img_path, title=f"Ground Truth - {text}")
        
        # Визуализация выходных данных
        output_img_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{i+1}_output.png")
        visualize_voxel_grid(output_voxel, save_path=output_img_path, title=f"Prediction - {text}")
        
        # 3D визуализация (если доступно)
        try:
            input_obj_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{i+1}_input.obj")
            create_3d_visualization(input_voxel, save_path=input_obj_path)
            
            output_obj_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{i+1}_output.obj")
            create_3d_visualization(output_voxel, threshold=0.5, save_path=output_obj_path)
        except Exception as e:
            logging.warning(f"Не удалось создать 3D визуализацию: {str(e)}")
    
    logging.info(f"Визуализации сохранены в {output_dir}")

def log_metrics(metrics, epoch, tensorboard_writer=None):
    """
    Логирование метрик в различные системы (Weights & Biases, TensorBoard).
    """
    # Логирование в Weights & Biases
    if wandb.run is not None:
        wandb.log(metrics, step=epoch)
    
    # Логирование в TensorBoard
    if tensorboard_writer is not None:
        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, epoch)

def main(args):
    """
    Main function for model training.
    """
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Получение имени эксперимента
    experiment_name = getattr(config, 'experiment_name', 'text_to_3d')
    if args.name:
        experiment_name = args.name
    
    # Настройка логирования
    logger = setup_logging(config.log_dir, experiment_name)
    logger.info(f"Starting model training for experiment: {experiment_name}")
    
    # Настройка среды обучения
    device = setup_training_environment(config)
    logger.info(f"Using device: {device}")
    
    # Инициализация систем логирования
    tensorboard_writer = init_logging_platforms(config, experiment_name)
    
    # Создание датасетов и загрузчиков данных
    train_loader, val_loader = create_dataloaders(config)
    
    # Создание модели
    model = create_model(config)
    model = model.to(device)
    
    # Создание функции потерь
    loss_fn = create_loss_function(config)
    
    # Создание оптимизатора
    optimizer = create_optimizer(model, config)
    
    # Создание планировщика скорости обучения
    lr_scheduler = create_lr_scheduler(optimizer, config)
    
    # Загрузка чекпоинта, если указан
    start_epoch = 0
    best_val_iou = 0.0
    if args.checkpoint:
        checkpoint_info = model.load_checkpoint(args.checkpoint, optimizer)
        if 'epoch' in checkpoint_info:
            start_epoch = checkpoint_info['epoch'] + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        if 'metrics' in checkpoint_info and 'val_iou' in checkpoint_info['metrics']:
            best_val_iou = checkpoint_info['metrics']['val_iou']
            logger.info(f"Best validation IoU from checkpoint: {best_val_iou:.4f}")
    
    # Параметры для ранней остановки
    if hasattr(config.training, 'early_stopping'):
        early_stopping_patience = getattr(config.training.early_stopping, 'patience', float('inf'))
        early_stopping_min_delta = getattr(config.training.early_stopping, 'min_delta', 0.0)
    else:
        early_stopping_patience = float('inf')
        early_stopping_min_delta = 0.0
    early_stopping_counter = 0
    
    # Интервалы сохранения и оценки
    save_interval = getattr(config.training, 'save_interval', 1)
    evaluate_interval = getattr(config.training, 'evaluate_interval', 1)
    
    # Основной цикл обучения
    for epoch in range(start_epoch, config.training.num_epochs):
        # Обучение на одной эпохе
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config)
        
        # Обновление планировщика скорости обучения
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Валидация модели
        if (epoch + 1) % evaluate_interval == 0:
            val_loss, val_iou, val_f_score, visualization_data = validate(model, val_loader, loss_fn, device, epoch, config)
            
            # Создание визуализаций
            visualization_dir = os.path.join(config.log_dir, experiment_name, 'visualizations')
            save_visualizations(visualization_data, epoch, visualization_dir)
            
            # Логирование метрик
            metrics = {
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_f_score': val_f_score,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            log_metrics(metrics, epoch, tensorboard_writer)
            
            # Сохранение лучшей модели
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_model_path = os.path.join(config.checkpoint_dir, f"{experiment_name}_best.pt")
                model.save_checkpoint(
                    best_model_path,
                    epoch=epoch,
                    metrics={'val_iou': val_iou, 'val_f_score': val_f_score},
                    optimizer=optimizer
                )
                logger.info(f"Сохранена лучшая модель с val_iou: {val_iou:.4f}")
                
                # Сброс счетчика ранней остановки
                early_stopping_counter = 0
            else:
                # Увеличение счетчика ранней остановки
                early_stopping_counter += 1
                logger.info(f"Ранняя остановка: {early_stopping_counter}/{early_stopping_patience}")
                
                # Проверка условия ранней остановки
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Ранняя остановка после {epoch+1} эпох из-за отсутствия улучшения валидационного IoU")
                    break
        
        # Периодическое сохранение чекпоинта
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"{experiment_name}_epoch_{epoch+1}.pt")
            model.save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                metrics={'train_loss': train_loss, 'train_iou': train_iou},
                optimizer=optimizer
            )
            logger.info(f"Сохранен чекпоинт для эпохи {epoch+1}")
    
    # Сохранение финальной модели
    final_model_path = os.path.join(config.checkpoint_dir, f"{experiment_name}_final.pt")
    model.save_checkpoint(
        final_model_path,
        epoch=config.training.num_epochs - 1,
        metrics={'train_loss': train_loss, 'train_iou': train_iou},
        optimizer=optimizer
    )
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    # Закрытие tensorboard writer
    if tensorboard_writer is not None:
        tensorboard_writer.close()
    
    # Завершение wandb сессии
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model for generating 3D objects from text descriptions")
    parser.add_argument("--config", type=str, default="modelit/configs/enhanced_architecture.yaml", help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()
    
    main(args) 