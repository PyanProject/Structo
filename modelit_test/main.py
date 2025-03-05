#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Главный файл приложения ModelIT для генерации 3D моделей из текстовых описаний.
Поддерживает как веб-интерфейс, так и обучение модели.
"""

import os
import sys
import json
import argparse
import logging
import datetime
import threading
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Импортируем компоненты модели
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelit_test.src.generate import Generator
from modelit_test.src.train import Trainer
from modelit_test.configs.default import ModelConfig

app = Flask(__name__)
app.secret_key = "modelit-secret-key-2024"
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join('static', 'output')

# Глобальные переменные
generator = None
model_config = None
generation_stats = {
    "total_generated": 0,
    "last_generation_time": None,
    "avg_generation_time": 0,
}

def sanitize_filename(prompt: str) -> str:
    """
    Очищает текст запроса для использования в имени файла.
    
    Args:
        prompt: Исходный текст запроса.
        
    Returns:
        Очищенный текст, пригодный для использования в имени файла.
    """
    # Удаляем недопустимые символы
    s = re.sub(r'[\\/*?:"<>|]', "", prompt)
    # Ограничиваем длину
    s = s.strip()[:50]
    # Если строка пустая, используем временную метку
    if not s:
        s = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return s

def initialize_app(args):
    """
    Инициализирует приложение и загружает модель.
    
    Args:
        args: Аргументы командной строки.
    """
    global generator, model_config
    
    # Создаем директории, если они не существуют
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Загружаем конфигурацию
    model_config = ModelConfig()
    
    # Инициализируем генератор, если путь к чекпойнту указан
    if args.checkpoint:
        logging.info(f"Загрузка модели из чекпойнта: {args.checkpoint}")
        try:
            generator = Generator(checkpoint_path=args.checkpoint)
            logging.info("Модель успешно загружена")
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            generator = None
    else:
        logging.warning("Чекпойнт не указан, генерация моделей недоступна")

@app.route('/')
def index():
    """Главная страница."""
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    """Страница генерации модели."""
    global generator, generation_stats
    
    if request.method == 'POST':
        # Получаем параметры из запроса
        prompt = request.form.get('prompt', '').strip()
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        batch_size = int(request.form.get('batch_size', 1))
        save_mesh = request.form.get('save_mesh', 'on') == 'on'
        save_preview = request.form.get('save_preview', 'on') == 'on'
        
        if not prompt:
            return jsonify({"error": "Необходимо указать описание модели"}), 400
        
        if not generator:
            return jsonify({"error": "Модель не загружена"}), 500
        
        # Имя файла на основе запроса
        filename_base = sanitize_filename(prompt)
        output_dir = os.path.join(app.root_path, app.config['OUTPUT_FOLDER'], filename_base)
        os.makedirs(output_dir, exist_ok=True)
        
        # Генерируем модель в отдельном потоке
        def generate_model_thread():
            try:
                start_time = datetime.datetime.now()
                
                # Генерируем модель
                mesh_path, preview_path = generator.generate(
                    prompt=prompt,
                    output_dir=output_dir,
                    guidance_scale=guidance_scale,
                    batch_size=batch_size,
                    save_mesh=save_mesh,
                    save_preview=save_preview
                )
                
                # Относительные пути для веб-интерфейса
                rel_mesh_path = os.path.relpath(mesh_path, os.path.join(app.root_path, 'static'))
                rel_preview_path = os.path.relpath(preview_path, os.path.join(app.root_path, 'static'))
                
                # Обновляем статистику
                end_time = datetime.datetime.now()
                generation_time = (end_time - start_time).total_seconds()
                
                generation_stats["total_generated"] += 1
                generation_stats["last_generation_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Обновляем среднее время
                if generation_stats["avg_generation_time"] == 0:
                    generation_stats["avg_generation_time"] = generation_time
                else:
                    generation_stats["avg_generation_time"] = (
                        generation_stats["avg_generation_time"] * (generation_stats["total_generated"] - 1) +
                        generation_time
                    ) / generation_stats["total_generated"]
                
                # Сохраняем метаданные
                metadata = {
                    "prompt": prompt,
                    "guidance_scale": guidance_scale,
                    "batch_size": batch_size,
                    "mesh_file": os.path.basename(mesh_path),
                    "preview_file": os.path.basename(preview_path),
                    "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "generation_time": generation_time
                }
                
                with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Модель успешно сгенерирована: {prompt}")
                
            except Exception as e:
                logging.error(f"Ошибка при генерации модели: {e}")
        
        # Запускаем генерацию в отдельном потоке
        thread = threading.Thread(target=generate_model_thread)
        thread.start()
        
        # Возвращаем успешный ответ
        return redirect(url_for('generate'))
    
    # Для GET запроса просто отображаем страницу
    return render_template('generate.html')

@app.route('/train')
def train_page():
    """Страница с информацией об обучении."""
    return render_template('train.html')

@app.route('/info')
def info():
    """Страница с информацией о системе."""
    # Собираем информацию о системе
    system_info = {
        "os": os.name,
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_gb": round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 2),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": generator is not None,
        "stats": generation_stats,
        "dependencies": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "flask": Flask.__version__,
            "trimesh": "Unknown"
        }
    }
    
    try:
        import trimesh
        system_info["dependencies"]["trimesh"] = trimesh.__version__
    except ImportError:
        pass
    
    # Добавляем информацию о CUDA, если доступно
    if system_info["cuda_available"]:
        system_info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
            "gpu_memory": round(torch.cuda.get_device_properties(0).total_memory / (1024.**3), 2)
                         if torch.cuda.device_count() > 0 else 0
        })
    
    # Добавляем информацию о чекпойнте
    if generator:
        system_info["checkpoint_path"] = generator.checkpoint_path
        checkpoint_size = os.path.getsize(generator.checkpoint_path) / (1024.**2)
        system_info["model_size"] = round(checkpoint_size, 2)
    else:
        system_info["checkpoint_path"] = "Не загружен"
        system_info["model_size"] = 0
    
    return render_template('info.html', info=system_info)

def run_training(args):
    """
    Запускает обучение модели.
    
    Args:
        args: Аргументы командной строки.
    """
    # Инициализируем тренер
    trainer = Trainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        resume_from=args.resume_from,
        no_wandb=args.no_wandb
    )
    
    # Запускаем обучение
    trainer.train()
    
    logging.info("Обучение завершено!")

def main():
    """Главная функция приложения."""
    parser = argparse.ArgumentParser(description="ModelIT - Генерация 3D моделей из текста")
    
    # Общие аргументы
    parser.add_argument("--checkpoint", type=str, help="Путь к чекпойнту модели")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Хост для запуска веб-интерфейса")
    parser.add_argument("--port", type=int, default=5000, help="Порт для запуска веб-интерфейса")
    parser.add_argument("--debug", action="store_true", help="Запуск в режиме отладки")
    
    # Аргументы для обучения
    parser.add_argument("--train", action="store_true", help="Запуск обучения вместо веб-интерфейса")
    parser.add_argument("--data_dir", type=str, help="Директория с данными для обучения")
    parser.add_argument("--output_dir", type=str, help="Директория для сохранения результатов обучения")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча для обучения")
    parser.add_argument("--num_epochs", type=int, default=50, help="Количество эпох обучения")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Скорость обучения")
    parser.add_argument("--resume_from", type=str, help="Путь к чекпойнту для продолжения обучения")
    parser.add_argument("--no_wandb", action="store_true", help="Отключить логирование в Weights & Biases")
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Запускаем обучение или веб-интерфейс
    if args.train:
        if not args.data_dir or not args.output_dir:
            logging.error("Для обучения необходимо указать --data_dir и --output_dir")
            return
        
        run_training(args)
    else:
        # Инициализируем приложение
        initialize_app(args)
        
        # Запускаем веб-интерфейс
        app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 