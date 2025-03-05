#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для обучения 3D-генеративной модели на датасете Objaverse.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm

# Импортируем компоненты нашей модели
from .data import create_dataloader
from .models import Generator, TextEncoder, create_model
from .utils import setup_logging

# Пытаемся импортировать wandb для логирования
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb не установлен. Логирование будет только локальным.")

class Trainer:
    """
    Класс для обучения 3D-генеративной модели.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: Dict[str, Any],
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        resume_from: Optional[str] = None,
        device: Optional[str] = None,
        mixed_precision: bool = True,
        save_interval: int = 1000,
        eval_interval: int = 500,
        log_interval: int = 100,
        no_wandb: bool = False,
        project_name: str = "modelit",
        experiment_name: Optional[str] = None,
    ):
        """
        Инициализация тренера.
        
        Args:
            data_dir: Директория с данными.
            output_dir: Директория для сохранения чекпоинтов и логов.
            config: Конфигурация модели.
            batch_size: Размер батча.
            learning_rate: Скорость обучения.
            num_epochs: Количество эпох.
            resume_from: Путь к чекпоинту для продолжения обучения.
            device: Устройство для обучения ('cuda', 'cpu').
            mixed_precision: Использовать ли смешанную точность.
            save_interval: Интервал сохранения чекпоинтов (в шагах).
            eval_interval: Интервал оценки модели (в шагах).
            log_interval: Интервал логирования (в шагах).
            no_wandb: Отключить логирование в wandb.
            project_name: Название проекта в wandb.
            experiment_name: Название эксперимента в wandb.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.resume_from = resume_from
        self.mixed_precision = mixed_precision
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.use_wandb = WANDB_AVAILABLE and not no_wandb
        self.project_name = project_name
        
        # Создаем директории для выходных данных
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Определяем устройство
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Настраиваем логирование
        self.logger = self.setup_logging()
        
        # Создаем модель
        self.model = self.init_model()
        
        # Настраиваем оптимизатор
        self.optimizer = self.setup_optimizer()
        
        # Настраиваем скейлер для смешанной точности
        self.scaler = GradScaler() if self.mixed_precision and torch.cuda.is_available() else None
        
        # Счетчики для обучения
        self.current_epoch = 0
        self.global_step = 0
        
        # Лучшее значение метрики
        self.best_val_loss = float('inf')
        
        # Настраиваем Weights & Biases
        if self.use_wandb:
            self.setup_wandb(experiment_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
        # Восстанавливаем из чекпоинта, если указан
        if self.resume_from:
            self.load_checkpoint(self.resume_from)
    
    def setup_logging(self) -> logging.Logger:
        """
        Настройка логирования.
        
        Returns:
            Настроенный логгер.
        """
        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)
        
        # Форматировщик для логов
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Обработчик для файла
        log_file = self.log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def setup_wandb(self, run_name: str):
        """
        Настройка Weights & Biases.
        
        Args:
            run_name: Название запуска.
        """
        try:
            wandb.init(
                project=self.project_name,
                name=run_name,
                config={
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "device": str(self.device),
                    "model_config": self.config,
                }
            )
            # Логируем модель
            wandb.watch(self.model)
        except Exception as e:
            self.logger.error(f"Ошибка инициализации wandb: {e}")
            self.use_wandb = False
    
    def init_model(self) -> nn.Module:
        """
        Инициализация модели.
        
        Returns:
            Инициализированная модель.
        """
        model = create_model(self.config)
        model = model.to(self.device)
        
        self.logger.info(f"Модель инициализирована на устройстве {self.device}")
        self.logger.info(f"Количество параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def setup_optimizer(self) -> optim.Optimizer:
        """
        Настройка оптимизатора.
        
        Returns:
            Настроенный оптимизатор.
        """
        # Настраиваем оптимизатор (Adam с параметрами по умолчанию)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.get("weight_decay", 0.0)
        )
        
        return optimizer
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """
        Сохранение чекпоинта.
        
        Args:
            filename: Имя файла чекпоинта.
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Загрузка чекпоинта.
        
        Args:
            checkpoint_path: Путь к файлу чекпоинта.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.error(f"Чекпоинт не найден: {checkpoint_path}")
            return
        
        self.logger.info(f"Загрузка чекпоинта: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Загружаем состояние модели
        self.model.load_state_dict(checkpoint["model"])
        
        # Загружаем состояние оптимизатора
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Восстанавливаем счетчики
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        # Восстанавливаем скейлер, если он есть
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        self.logger.info(f"Чекпоинт загружен, эпоха: {self.current_epoch}, шаг: {self.global_step}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Выполнение одного шага обучения.
        
        Args:
            batch: Батч данных.
            
        Returns:
            Словарь с метриками.
        """
        # Перемещаем данные на устройство
        voxels = batch["voxels"].to(self.device)
        views = batch["views"].to(self.device) if "views" in batch else None
        captions = batch["caption"]
        
        # Обнуляем градиенты
        self.optimizer.zero_grad()
        
        # Вычисляем потери с учетом смешанной точности
        if self.mixed_precision and torch.cuda.is_available():
            with autocast():
                loss_dict = self.model(voxels, captions, views)
                total_loss = sum(loss_dict.values())
            
            # Выполняем обратное распространение с использованием скейлера
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Обычный проход без смешанной точности
            loss_dict = self.model(voxels, captions, views)
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            self.optimizer.step()
        
        # Собираем метрики
        metrics = {name: loss.item() for name, loss in loss_dict.items()}
        metrics["total_loss"] = total_loss.item()
        
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Оценка модели на валидационных данных.
        
        Args:
            dataloader: Загрузчик данных для валидации.
            
        Returns:
            Словарь с метриками.
        """
        self.model.eval()
        total_metrics = {}
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Оценка", leave=False):
                # Перемещаем данные на устройство
                voxels = batch["voxels"].to(self.device)
                views = batch["views"].to(self.device) if "views" in batch else None
                captions = batch["caption"]
                
                # Вычисляем потери
                with autocast() if self.mixed_precision and torch.cuda.is_available() else nullcontext():
                    loss_dict = self.model(voxels, captions, views)
                
                # Собираем метрики
                batch_count += 1
                for name, loss in loss_dict.items():
                    total_metrics[name] = total_metrics.get(name, 0.0) + loss.item()
        
        # Вычисляем средние значения метрик
        avg_metrics = {f"val_{name}": value / batch_count for name, value in total_metrics.items()}
        avg_metrics["val_total_loss"] = sum(avg_metrics.values())
        
        self.model.train()
        return avg_metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Логирование метрик.
        
        Args:
            metrics: Словарь с метриками.
            step: Текущий шаг.
            prefix: Префикс для названий метрик.
        """
        # Логируем в консоль
        self.logger.info(f"{prefix} шаг {step}: " + ", ".join([f"{name}: {value:.4f}" for name, value in metrics.items()]))
        
        # Логируем в wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def train(self):
        """
        Процесс обучения модели.
        """
        self.logger.info(f"Начинаем обучение на {self.device}, смешанная точность: {self.mixed_precision}")
        
        # Создаем загрузчики данных
        train_loader = create_dataloader(
            data_dir=self.data_dir,
            split="train",
            batch_size=self.batch_size,
            num_workers=4,
            voxel_size=self.config.get("voxel_size", 64),
            is_objaverse=True,
            cache_path=str(self.data_dir / "cache"),
        )
        
        val_loader = create_dataloader(
            data_dir=self.data_dir,
            split="val",
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            voxel_size=self.config.get("voxel_size", 64),
            is_objaverse=True,
            cache_path=str(self.data_dir / "cache"),
        )
        
        self.logger.info(f"Загружено {len(train_loader.dataset)} образцов для обучения и {len(val_loader.dataset)} для валидации")
        
        # Переводим модель в режим обучения
        self.model.train()
        
        # Начинаем с текущей эпохи (может быть > 0, если загружен чекпоинт)
        start_epoch = self.current_epoch
        
        # Основной цикл обучения
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Эпоха {epoch+1}/{self.num_epochs}")
            
            epoch_metrics = {}
            epoch_start_time = time.time()
            
            # Цикл по батчам
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Эпоха {epoch+1}")):
                # Выполняем шаг обучения
                metrics = self.train_step(batch)
                
                # Обновляем глобальный счетчик шагов
                self.global_step += 1
                
                # Собираем метрики за эпоху
                for name, value in metrics.items():
                    epoch_metrics[name] = epoch_metrics.get(name, 0.0) + value
                
                # Логируем метрики с указанным интервалом
                if self.global_step % self.log_interval == 0:
                    self.log_metrics(metrics, self.global_step, prefix="Обучение")
                
                # Оцениваем модель на валидационных данных с указанным интервалом
                if self.global_step % self.eval_interval == 0:
                    val_metrics = self.evaluate(val_loader)
                    self.log_metrics(val_metrics, self.global_step, prefix="Валидация")
                    
                    # Сохраняем лучшую модель
                    if val_metrics["val_total_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_total_loss"]
                        self.save_checkpoint("best_model.pt")
                
                # Сохраняем чекпоинт с указанным интервалом
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
            
            # Вычисляем средние метрики за эпоху
            avg_epoch_metrics = {name: value / len(train_loader) for name, value in epoch_metrics.items()}
            
            # Логируем метрики за эпоху
            self.logger.info(f"Эпоха {epoch+1} завершена за {time.time() - epoch_start_time:.2f} сек.")
            self.log_metrics(avg_epoch_metrics, self.global_step, prefix=f"Эпоха {epoch+1}")
            
            # Сохраняем чекпоинт в конце каждой эпохи
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Также сохраняем как последний чекпоинт
            self.save_checkpoint("checkpoint_latest.pt")
        
        # Сохраняем финальную модель
        self.save_checkpoint("checkpoint_final.pt")
        self.logger.info("Обучение завершено!")
        
        # Завершаем wandb
        if self.use_wandb:
            wandb.finish()

# Контекстный менеджер для отключения вывода
class nullcontext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логирования.
    
    Args:
        log_dir: Директория для сохранения логов.
        log_level: Уровень логирования.
        
    Returns:
        Настроенный логгер.
    """
    # Создаем директорию для логов
    os.makedirs(log_dir, exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger("modelit")
    logger.setLevel(log_level)
    
    # Форматировщик для логов
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Обработчик для файла
    log_file = os.path.join(log_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из файла.
    
    Args:
        config_file: Путь к файлу конфигурации.
        
    Returns:
        Словарь с конфигурацией.
    """
    # Проверяем, что файл существует
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_file}")
    
    # Определяем формат файла по расширению
    if config_file.endswith(".py"):
        # Загружаем как Python-модуль
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Создаем словарь конфигурации из атрибутов модуля
        config = {}
        for key in dir(config_module):
            if not key.startswith("__"):
                config[key] = getattr(config_module, key)
    
    elif config_file.endswith(".json"):
        # Загружаем как JSON-файл
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    else:
        raise ValueError(f"Неподдерживаемый формат файла конфигурации: {config_file}")
    
    return config

def main():
    """
    Главная функция для запуска обучения.
    """
    parser = argparse.ArgumentParser(description="Обучение 3D-генеративной модели")
    parser.add_argument("--config", type=str, required=True, help="Путь к файлу конфигурации")
    parser.add_argument("--data_dir", type=str, required=True, help="Директория с данными")
    parser.add_argument("--output_dir", type=str, required=True, help="Директория для сохранения чекпоинтов и логов")
    parser.add_argument("--log_dir", type=str, help="Директория для сохранения логов (если отличается от output_dir)")
    parser.add_argument("--resume_from", type=str, help="Путь к чекпоинту для продолжения обучения")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Скорость обучения")
    parser.add_argument("--num_epochs", type=int, default=100, help="Количество эпох")
    parser.add_argument("--device", type=str, help="Устройство для обучения ('cuda', 'cpu')")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Отключить смешанную точность")
    parser.add_argument("--no_wandb", action="store_true", help="Отключить логирование в wandb")
    parser.add_argument("--project_name", type=str, default="modelit", help="Название проекта в wandb")
    parser.add_argument("--experiment_name", type=str, help="Название эксперимента в wandb")
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    logger = setup_logging(log_dir)
    
    try:
        # Загружаем конфигурацию
        logger.info(f"Загрузка конфигурации из {args.config}")
        config = load_config(args.config)
        
        # Создаем и запускаем тренер
        trainer = Trainer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            resume_from=args.resume_from,
            device=args.device,
            mixed_precision=not args.no_mixed_precision,
            no_wandb=args.no_wandb,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
        )
        
        # Запускаем обучение
        trainer.train()
        
    except Exception as e:
        logger.exception(f"Произошла ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 