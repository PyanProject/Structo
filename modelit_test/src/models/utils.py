#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .transformer import PointDiffusionTransformer, TextConditionedTransformer

logger = logging.getLogger(__name__)

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Загружает конфигурационный файл
    
    Args:
        config_name: имя конфигурационного файла
        
    Returns:
        Словарь с параметрами конфигурации
    """
    # По умолчанию ищем конфигурации в папке configs
    config_dir = Path("configs")
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        # Если нет файла в configs, используем встроенную конфигурацию по умолчанию
        logger.warning(f"Конфигурационный файл {config_path} не найден, используется конфигурация по умолчанию")
        
        if config_name == "diffusion":
            # Конфигурация для диффузии по умолчанию
            return {
                "diffusion": {
                    "schedule": "linear",
                    "num_timesteps": 1000,
                    "beta_start": 1e-4,
                    "beta_end": 2e-2
                },
                "model_mean_type": "epsilon",
                "model_var_type": "fixed_small",
                "loss_type": "mse"
            }
        elif config_name.startswith("text"):
            # Конфигурация для текстовой модели по умолчанию
            return {
                "model_type": config_name,
                "input_channels": 3,
                "output_channels": 3,
                "n_ctx": 4096,
                "width": 512,
                "layers": 12,
                "heads": 8,
                "dropout": 0.1,
                "text_embed_dim": 512
            }
        elif config_name.startswith("image"):
            # Конфигурация для модели изображений по умолчанию
            return {
                "model_type": config_name,
                "input_channels": 3,
                "output_channels": 3,
                "n_ctx": 4096,
                "width": 512,
                "layers": 12,
                "heads": 8,
                "dropout": 0.1
            }
        else:
            raise ValueError(f"Неизвестное имя конфигурации: {config_name}")
    
    # Если файл существует, загружаем его
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """
    Загружает предобученную модель
    
    Args:
        model_name: имя модели
        device: устройство для загрузки модели
        
    Returns:
        Загруженная модель
    """
    # Загружаем конфигурацию модели
    config = load_config(model_name)
    
    # Создаем модель на основе типа
    if model_name.startswith("text"):
        model = TextConditionedTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=config.get("input_channels", 3),
            output_channels=config.get("output_channels", 3),
            n_ctx=config.get("n_ctx", 4096),
            width=config.get("width", 512),
            layers=config.get("layers", 12),
            heads=config.get("heads", 8),
            dropout=config.get("dropout", 0.1),
            text_embed_dim=config.get("text_embed_dim", 512)
        )
    elif model_name.startswith("image"):
        model = PointDiffusionTransformer(
            device=device,
            dtype=torch.float32,
            input_channels=config.get("input_channels", 3),
            output_channels=config.get("output_channels", 3),
            n_ctx=config.get("n_ctx", 4096),
            width=config.get("width", 512),
            layers=config.get("layers", 12),
            heads=config.get("heads", 8),
            dropout=config.get("dropout", 0.1)
        )
    else:
        raise ValueError(f"Неизвестное имя модели: {model_name}")
    
    # Пытаемся загрузить веса модели
    try:
        checkpoint_path = Path("models") / f"{model_name}.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Модель загружена из {checkpoint_path}")
        else:
            logger.warning(f"Веса модели не найдены в {checkpoint_path}, используется случайная инициализация")
    except Exception as e:
        logger.error(f"Ошибка при загрузке весов модели: {e}")
        logger.warning("Используется случайная инициализация")
    
    model.eval()  # Устанавливаем режим оценки
    return model

def save_model(model: torch.nn.Module, model_name: str, save_dir: str = "models", **kwargs) -> str:
    """
    Сохраняет модель в файл
    
    Args:
        model: модель для сохранения
        model_name: имя модели
        save_dir: директория для сохранения
        **kwargs: дополнительные данные для сохранения в checkpoint
        
    Returns:
        Путь к сохраненной модели
    """
    # Создаем директорию, если её нет
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Формируем путь к файлу
    model_path = save_path / f"{model_name}.pth"
    
    # Создаем словарь с данными
    checkpoint = {
        "model_state_dict": model.state_dict(),
        **kwargs
    }
    
    # Сохраняем модель
    torch.save(checkpoint, model_path)
    logger.info(f"Модель сохранена в {model_path}")
    
    return str(model_path)

def get_available_models(models_dir: str = "models") -> Dict[str, Dict[str, Any]]:
    """
    Получает список доступных предобученных моделей
    
    Args:
        models_dir: директория с моделями
        
    Returns:
        Словарь с информацией о доступных моделях
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return {}
    
    models = {}
    for model_path in models_path.glob("*.pth"):
        model_name = model_path.stem
        
        # Загружаем метаданные модели
        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
            models[model_name] = metadata
        except:
            models[model_name] = {"note": "Метаданные недоступны"}
    
    return models 