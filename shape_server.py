import os
import json
import time
import torch
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import threading
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
import numpy as np
import cv2
from PIL import Image
import io

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True
    }
})  # Разрешаем кросс-доменные запросы

# Глобальные переменные для моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_loaded = False
text_model = None
diffusion = None
xm = None

# Глобальные переменные для отслеживания прогресса
current_generation_progress = {
    'prompt': '',
    'status': 'idle',  # idle, loading_models, generating, completed, error
    'progress': 0,
    'message': '',
    'start_time': 0,
    'end_time': 0
}

# Функции из test_shape_e.py
def print_gpu_memory(message=""):
    """Выводит информацию о доступной памяти GPU"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_memory = total_memory - reserved_memory
        
        # Обновляем информацию о памяти в мониторинге
        if message:
            current_generation_progress['message'] = message
        
        print(f"{message} GPU Memory: Total={total_memory:.2f}GB, "
              f"Reserved={reserved_memory:.2f}GB, Allocated={allocated_memory:.2f}GB, "
              f"Free={free_memory:.2f}GB")
    else:
        print("CUDA недоступен")

def check_gpu():
    """Проверяет доступность GPU"""
    if not torch.cuda.is_available():
        print("CUDA недоступен. Используется CPU, что делает генерацию очень медленной.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    print(f"Используется GPU: {device_name}")
    print_gpu_memory("Начальное")
    return True

def sanitize_filename(prompt):
    """Очищает строку промпта для использования в имени файла"""
    invalid_chars = '<>:"/\\|?*'
    filename = prompt
    for char in invalid_chars:
        filename = filename.replace(char, '')
    # Ограничиваем длину имени файла
    return filename[:50].strip()

def load_models():
    """Загружает все необходимые модели для генерации"""
    global model_loaded, xm, diffusion
    
    try:
        print("Загрузка моделей для генерации...")
        
        # Определяем устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загружаем модели
        xm = load_model('transmitter', device=device)
        text_model = load_model('text300M', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        
        # Устанавливаем флаг, что модели загружены
        model_loaded = True
        print("Модели успешно загружены")
        return True
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {str(e)}")
        model_loaded = False
        return False

def generate_model(prompt, output_path, subscription):
    """Генерирует 3D модель на основе текстового описания"""
    global current_generation_progress
    
    try:
        print("\n=== Начало генерации модели ===")
        print(f"Промпт: {prompt}")
        print(f"Путь для сохранения: {output_path}")
        
        current_generation_progress.update({
            'status': 'generating',
            'progress': 0,
            'message': 'Начало генерации...',
            'start_time': time.time()
        })
        
        # Проверяем, что промпт не пустой
        if not prompt or len(prompt.strip()) == 0:
            print("Ошибка: промпт не может быть пустым")
            current_generation_progress.update({
                'status': 'error',
                'message': 'Промпт не может быть пустым'
            })
            return False
        
        # Проверяем GPU доступность
        if not check_gpu():
            print("Ошибка: GPU недоступен")
            current_generation_progress.update({
                'status': 'error',
                'message': 'GPU недоступен'
            })
            return False
        
        # Загружаем модели, если они еще не загружены
        if not model_loaded:
            print("Загрузка моделей...")
            current_generation_progress.update({
                'status': 'loading_models',
                'message': 'Загрузка моделей...',
                'progress': 10
            })
            load_models()
        
        print_gpu_memory("Модели успешно загружены")
        current_generation_progress.update({
            'status': 'generating',
            'message': 'Генерация латентных представлений...',
            'progress': 20
        })
        
        # Создаем конфигурацию диффузии
        batch_size = 1
        guidance_scale = 15.0
        
        # Получаем модель text300M для генерации латентных представлений
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Загрузка text300M модели на устройство {device}...")
        text_model = load_model('text300M', device=device)
        print("text300M модель загружена")
        
        # Генерируем латентные представления
        print("Начало генерации латентных представлений...")
        current_generation_progress.update({
            'status': 'generating',
            'message': 'Генерация модели...',
            'progress': 30
        })

        if subscription == 1:  # Free
            guidance_scale = 7.5
            karras_steps = 8
            sigma_min = 1e-2
            sigma_max = 80
            batch_size = 1
        elif subscription == 2:  # Pro
            guidance_scale = 10.0
            karras_steps = 32
            sigma_min = 1e-3
            sigma_max = 120
            batch_size = 2
        elif subscription == 3:  # Business
            guidance_scale = 10.0
            karras_steps = 32
            sigma_min = 1e-3
            sigma_max = 120
            batch_size = 2
        
        latents = sample_latents(
            batch_size=batch_size,
            model=text_model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=karras_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s_churn=0
        )
        print("Латентные представления сгенерированы")
        current_generation_progress.update({
            'status': 'generating',
            'message': 'Создание меша...',
            'progress': 80
        })
        
        print("Создание меша из латентных представлений...")
        mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
        print("Меш создан")
        
        current_generation_progress.update({
            'status': 'generating',
            'message': 'Сохранение модели...',
            'progress': 90
        })
        
        print(f"Сохранение модели в {output_path}")
        with open(output_path, 'w') as f:
            mesh.write_obj(f)
        print("Модель успешно сохранена")
        
        # Обновляем статус на завершено
        current_generation_progress.update({
            'status': 'completed',
            'message': 'Генерация завершена',
            'progress': 100,
            'end_time': time.time(),
            'file_path': output_path,
            'file_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
        })
        
        return True
    except Exception as e:
        print(f"\nОшибка при генерации модели: {str(e)}")
        print(f"Тип ошибки: {type(e).__name__}")
        import traceback
        print(f"Трейсбек: {traceback.format_exc()}")
        
        current_generation_progress.update({
            'status': 'error',
            'message': f'Ошибка: {str(e)}',
            'end_time': time.time()
        })
        
        return False

# Функция-колбэк для обновления прогресса
def update_progress_callback(info):
    """Обновляет информацию о прогрессе на основе колбэка диффузии"""
    global current_generation_progress
    
    try:
        # Получаем информацию о текущем шаге
        if hasattr(info, 'i') and hasattr(info, 'total'):
            step = info.i
            total_steps = info.total
            
            # Рассчитываем процент выполнения (от 15 до 80 процентов)
            # потому что до 15% идет загрузка, после 80% идет создание меша
            if total_steps > 0:
                # Преобразуем прогресс в диапазон 15-80%
                step_progress = (step / total_steps) * 65 + 15
                
                # Обновляем информацию о прогрессе
                current_generation_progress.update({
                    'progress': step_progress,
                    'message': f'Генерация модели: шаг {step}/{total_steps} ({int(step_progress)}%)'
                })
                
                # Выводим информацию в консоль каждые 10 шагов
                if step % 10 == 0 or step == total_steps - 1:
                    print(f"Прогресс генерации: {step}/{total_steps} ({int(step_progress)}%)")
    except Exception as e:
        print(f"Ошибка в колбэке прогресса: {str(e)}")
        # Игнорируем ошибки в колбэке, чтобы не прерывать основной процесс

# API эндпоинты
@app.route('/health', methods=['GET'])
def health_check():
    """Эндпоинт для проверки работоспособности сервера"""
    return jsonify({'status': 'ok', 'gpu_available': torch.cuda.is_available()})

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    """Эндпоинт для генерации 3D модели"""
    try:
        print("\n=== Получен запрос на генерацию модели ===")
        print(f"Заголовки запроса: {dict(request.headers)}")
        print(f"Метод запроса: {request.method}")
        print(f"Тип контента: {request.content_type}")
        print(f"Данные запроса: {request.get_data(as_text=True)}")
        
        if not request.is_json:
            error_msg = 'Ожидается application/json в Content-Type'
            print(f"Ошибка: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400

        try:
            data = request.get_json()
            print(f"Полученные JSON данные: {data}")
        except Exception as e:
            error_msg = f'Ошибка при разборе JSON: {str(e)}'
            print(f"Ошибка: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400

        prompt = data.get('prompt')
        subscription = data.get('subscription')
        print(f"Извлеченный промпт: {prompt}")

        if not prompt:
            error_msg = 'Не указан промпт'
            print(f"Ошибка: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400

        # Создаем имя файла из промпта
        safe_prompt = sanitize_filename(prompt)
        timestamp = int(time.time())
        filename = f"{safe_prompt}_{timestamp}.obj"
        output_path = os.path.join("output", filename)
        print(f"Подготовленный путь для сохранения: {output_path}")

        # Создаем директорию для вывода, если её нет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Директория для сохранения создана/проверена: {os.path.dirname(output_path)}")

        # Запускаем генерацию в отдельном потоке
        print("Запуск генерации в отдельном потоке...")
        thread = threading.Thread(
            target=generate_model,
            args=(prompt, output_path, subscription)
        )
        thread.daemon = True
        thread.start()
        print("Поток генерации запущен успешно")

        response_data = {
            'status': 'success',
            'message': 'Генерация модели запущена',
            'filename': filename
        }
        print(f"Отправляем ответ: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"Неожиданная ошибка при обработке запроса: {str(e)}"
        print(error_msg)
        import traceback
        print(f"Трейсбек: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

# Новый эндпоинт для получения текущего прогресса генерации
@app.route('/progress', methods=['GET'])
def get_progress():
    """Возвращает текущий прогресс генерации модели"""
    global current_generation_progress
    
    try:
        # Копируем текущий прогресс
        result = current_generation_progress.copy()
        
        # Добавляем дополнительную информацию
        if result['status'] in ['loading_models', 'generating']:
            if 'start_time' in result:
                elapsed_time = time.time() - result['start_time']
                result['elapsed_seconds'] = int(elapsed_time)
                
                # Оцениваем оставшееся время
                if result.get('progress', 0) > 0:
                    estimated_total = elapsed_time * 100 / result['progress']
                    result['remaining_seconds'] = int(estimated_total - elapsed_time)
                else:
                    result['remaining_seconds'] = -1
        
        # Проверяем наличие файла модели, если статус completed
        if result['status'] == 'completed':
            # Убедимся, что файл действительно существует и имеет размер
            output_dir = "output"
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                if files:
                    latest_file = max([os.path.join(output_dir, f) for f in files], key=os.path.getctime)
                    if os.path.exists(latest_file) and os.path.getsize(latest_file) > 0:
                        result['file_ready'] = True
                        result['file_size'] = os.path.getsize(latest_file)
                    else:
                        result['file_ready'] = False
        
        print(f"Текущий прогресс: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Ошибка при получении прогресса: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при получении прогресса: {str(e)}'
        }), 500

@app.route('/output/<filename>')
def serve_output_file(filename):
    """Отдает сгенерированный файл модели"""
    try:
        file_path = os.path.join('output', filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='model/obj')
        else:
            return jsonify({'error': 'Файл не найден'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Инициализация при запуске
if __name__ == '__main__':
    check_gpu()
    # Предварительная загрузка моделей
    threading.Thread(target=load_models).start()
    app.run(host='0.0.0.0', port=5678, debug=True) 