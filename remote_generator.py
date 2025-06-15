import os
import base64
import time
import json
import requests
import tempfile
from werkzeug.utils import secure_filename
from flask import jsonify, send_file
import threading

# Импортируем настройки из конфигурационного файла
from config import GENERATION_API_URL, HEALTH_CHECK_URL, REQUEST_TIMEOUT, PROGRESS_CHECK_URL, LOCAL_PC_IP, LOCAL_PC_PORT

# Используем собственную временную директорию вместо tempfile.gettempdir()
# Папка для временного хранения сгенерированных моделей
TEMP_MODEL_FOLDER = os.path.join('/project/tmp', 'structo_models')
os.makedirs(TEMP_MODEL_FOLDER, exist_ok=True)

def check_generator_status():
    """
    Проверка доступности генератора моделей на локальном ПК.
    
    Returns:
        dict: Статус генератора (или ошибка)
        int: HTTP код ответа
    """
    try:
        # Печатаем значения из конфигурации для отладки
        print(f"DEBUG: HEALTH_CHECK_URL = {HEALTH_CHECK_URL}")
        print(f"DEBUG: LOCAL_PC_IP = {LOCAL_PC_IP}")
        print(f"DEBUG: LOCAL_PC_PORT = {LOCAL_PC_PORT}")
        
        # Отправляем GET запрос к API для проверки доступности
        print(f"DEBUG: Отправка запроса на {HEALTH_CHECK_URL}")
        response = requests.get(HEALTH_CHECK_URL, timeout=10)
        print(f"DEBUG: Получен ответ со статусом {response.status_code}")
        
        # Проверяем успешность запроса
        if response.status_code == 200:
            return response.json(), 200
        else:
            # Возвращаем сообщение об ошибке и код ошибки
            print(f"DEBUG: Генератор недоступен, код ответа: {response.status_code}")
            return {
                'message': f'Генератор моделей недоступен (HTTP {response.status_code})',
                'status': 'error'
            }, response.status_code
            
    except requests.exceptions.ConnectionError as e:
        # Возвращаем сообщение о недоступности генератора
        print(f"DEBUG: Ошибка подключения: {str(e)}")
        return {
            'message': 'Не удалось подключиться к генератору моделей на локальном ПК. Возможно, он выключен или недоступен.',
            'status': 'error'
        }, 503
        
    except requests.exceptions.Timeout as e:
        # Возвращаем сообщение о таймауте запроса
        print(f"DEBUG: Таймаут: {str(e)}")
        return {
            'message': 'Таймаут при обращении к генератору моделей. Проверьте соединение с локальным ПК.',
            'status': 'error'
        }, 504
        
    except Exception as e:
        # Возвращаем информацию о любой другой ошибке
        return {
            'message': f'Ошибка при проверке статуса генератора: {str(e)}',
            'status': 'error'
        }, 500

def check_generation_progress():
    """
    Получение текущего прогресса генерации модели с локального ПК.
    
    Returns:
        dict: Информация о прогрессе генерации
        int: HTTP код ответа
    """
    try:
        # Отправляем запрос к API для получения текущего прогресса
        response = requests.get(PROGRESS_CHECK_URL, timeout=10)
        
        # Проверяем успешность запроса
        if response.status_code == 200:
            progress_data = response.json()
            return progress_data, 200
        else:
            # Возвращаем сообщение об ошибке и код ошибки
            return {
                'message': f'Ошибка при получении прогресса генерации (HTTP {response.status_code})',
                'status': 'error',
                'progress': 0
            }, response.status_code
            
    except requests.exceptions.ConnectionError:
        # Возвращаем сообщение о недоступности генератора
        return {
            'message': 'Не удалось подключиться к генератору моделей.',
            'status': 'error',
            'progress': 0
        }, 503
        
    except requests.exceptions.Timeout:
        # Возвращаем сообщение о таймауте запроса
        return {
            'message': 'Таймаут при обращении к генератору моделей.',
            'status': 'error',
            'progress': 0
        }, 504
        
    except Exception as e:
        # Возвращаем информацию о любой другой ошибке
        return {
            'message': f'Ошибка при получении прогресса: {str(e)}',
            'status': 'error',
            'progress': 0
        }, 500

def generate_model(prompt):
    """
    Запрашивает генерацию 3D-модели от локального сервера.
    
    Arguments:
        prompt (str): Текстовый промпт для генерации модели
        
    Returns:
        tuple: (dict с результатами генерации или ошибкой, HTTP код)
    """
    # Валидация промпта
    if not prompt or len(prompt.strip()) < 2:
        return {
            'success': False, 
            'error': 'Промпт должен содержать не менее 2 символов'
        }, 400
    
    print(f"=== Начало процесса генерации модели ===")
    print(f"Промпт: {prompt}")
    print(f"URL генерации: {GENERATION_API_URL}")
    print(f"DEBUG: Прямой вызов curl")
    
    print(f"Проверка доступности генератора...")
    # Проверяем, доступен ли генератор
    status, status_code = check_generator_status()
    print(f"DEBUG: Результат проверки генератора: {status}, код: {status_code}")
    
    if status_code != 200:
        print(f"Генератор недоступен: {status.get('message')}")
        return {
            'success': False, 
            'error': status.get('message', 'Генератор недоступен. Проверьте соединение с локальным ПК')
        }, status_code
    
    print(f"Генератор доступен, отправка запроса на генерацию...")
    
    # Санитизация промпта для использования в имени файла
    safe_prompt = ''.join(c if c.isalnum() else '_' for c in prompt.lower())
    
    # Ограничиваем длину имени файла
    if len(safe_prompt) > 50:
        safe_prompt = safe_prompt[:50]
    
    # Добавляем временную метку к имени файла
    timestamp = int(time.time())
    filename = f"{safe_prompt}_{timestamp}.obj"
    
    try:
        print(f"Подготовка данных для запроса:")
        print(f"- URL: {GENERATION_API_URL}")
        print(f"- Промпт: {prompt}")
        print(f"- Таймаут: {REQUEST_TIMEOUT} секунд")
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        print(f"DEBUG: Заголовки запроса: {headers}")
        print(f"DEBUG: Отправка тела запроса с промптом")
        
        response = requests.post(
            GENERATION_API_URL,
            json={'prompt': prompt},
            timeout=REQUEST_TIMEOUT,
            headers=headers
        )
        print(f"DEBUG: Ответ получен, статус: {response.status_code}")
        print(f"DEBUG: Получен ответ от сервера")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get('status') == 'success':
                    # URL для доступа к модели на странице
                    obj_url = f"/models/{filename}"
                    
                    # URL для скачивания модели
                    download_url = f"/download_model/{filename}"
                    
                    print(f"Запрос на генерацию успешно отправлен")
                    print(f"URL для просмотра модели: {obj_url}")
                    print(f"URL для скачивания: {download_url}")
                    
                    return {
                        'success': True,
                        'message': 'Запрос на генерацию модели отправлен',
                        'obj_url': obj_url,
                        'download_url': download_url
                    }, 200
                else:
                    error_msg = data.get('error', 'Неизвестная ошибка при генерации')
                    print(f"Ошибка генерации: {error_msg}")
                    return {
                        'success': False,
                        'error': error_msg
                    }, 500
            except json.JSONDecodeError as e:
                print(f"Ошибка при разборе JSON ответа: {str(e)}")
                print(f"Содержимое ответа: {response.text}")
                return {
                    'success': False,
                    'error': 'Ошибка при обработке ответа от сервера'
                }, 500
        else:
            error_msg = f'Ошибка сервера: {response.text}'
            print(f"Ошибка сервера: {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }, response.status_code
            
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения при запросе генерации:")
        print(f"- Тип ошибки: {type(e).__name__}")
        print(f"- Сообщение: {str(e)}")
        return {
            'success': False,
            'error': 'Ошибка соединения с генератором. Проверьте доступность локального ПК.'
        }, 503
    except Exception as e:
        print(f"Неожиданная ошибка при генерации:")
        print(f"- Тип ошибки: {type(e).__name__}")
        print(f"- Сообщение: {str(e)}")
        return {
            'success': False,
            'error': f'Неожиданная ошибка при генерации: {str(e)}'
        }, 500

def serve_model_file(filename):
    """Отдает файл модели для отображения на странице"""
    file_path = os.path.join(TEMP_MODEL_FOLDER, secure_filename(filename))
    
    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден. Пытаемся скопировать с локального сервера.")
        
        # Пытаемся скопировать файл с локального сервера
        if copy_model_from_local_server(filename):
            print(f"Файл успешно скопирован, отдаем его клиенту")
            return send_file(file_path, mimetype='model/obj')
        
        print(f"Не удалось скопировать файл. Создаем временную заглушку.")
        
        # Создаем простой временный файл в формате .obj
        temp_obj_content = """
# Временный OBJ файл
# Модель генерируется на локальном сервере
v -0.5 -0.5 -0.5
v -0.5 0.5 -0.5
v 0.5 0.5 -0.5
v 0.5 -0.5 -0.5
v -0.5 -0.5 0.5
v -0.5 0.5 0.5
v 0.5 0.5 0.5
v 0.5 -0.5 0.5
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
        with open(file_path, 'w') as f:
            f.write(temp_obj_content)
            
        return send_file(file_path, mimetype='model/obj')
    
    # Если файл существует, отдаем его
    return send_file(file_path, mimetype='model/obj')

def download_model_file(filename):
    """Отдает файл модели для скачивания"""
    file_path = os.path.join(TEMP_MODEL_FOLDER, secure_filename(filename))
    
    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден для скачивания. Создаем временную заглушку.")
        
        # Создаем простой временный файл в формате .obj с сообщением
        temp_obj_content = """
# Временный OBJ файл
# Модель все еще генерируется на локальном сервере
# Пожалуйста, подождите несколько минут и попробуйте снова
v -0.5 -0.5 -0.5
v -0.5 0.5 -0.5
v 0.5 0.5 -0.5
v 0.5 -0.5 -0.5
v -0.5 -0.5 0.5
v -0.5 0.5 0.5
v 0.5 0.5 0.5
v 0.5 -0.5 0.5
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
        """
        
        # Сохраняем временный файл
        try:
            with open(file_path, 'w') as f:
                f.write(temp_obj_content)
            print(f"Создана временная заглушка для скачивания в {file_path}")
        except Exception as e:
            print(f"Ошибка при создании временной заглушки для скачивания: {str(e)}")
            return {'success': False, 'error': 'Модель еще генерируется. Файл временно недоступен.'}, 202
    
    try:
        # Всегда возвращаем файл для скачивания, даже если это заглушка
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Ошибка при отправке файла для скачивания: {str(e)}")
        return {'success': False, 'error': str(e)}, 500

def copy_model_from_local_server(filename):
    """
    Копирует файл модели с локального сервера на VDS.
    
    Args:
        filename (str): Имя файла модели для копирования
        
    Returns:
        bool: True если копирование успешно, False в противном случае
    """
    try:
        # Формируем URL для загрузки файла с локального сервера
        url = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/output/{filename}"
        print(f"[COPY] Попытка скопировать файл с {url}")
        print(f"[COPY] Используем IP: {LOCAL_PC_IP}, порт: {LOCAL_PC_PORT}")
        
        # Отправляем GET запрос для получения файла
        print(f"[COPY] Отправляем GET запрос...")
        response = requests.get(url, timeout=30)
        
        print(f"[COPY] Получен ответ:")
        print(f"[COPY] - Статус код: {response.status_code}")
        print(f"[COPY] - Заголовки: {dict(response.headers)}")
        print(f"[COPY] - Размер ответа: {len(response.content)} байт")
        print(f"[COPY] - Тип контента: {response.headers.get('content-type', 'не указан')}")
        
        if response.status_code == 200:
            # Создаем путь для сохранения файла
            file_path = os.path.join(TEMP_MODEL_FOLDER, secure_filename(filename))
            print(f"[COPY] Сохраняем файл в {file_path}")
            
            # Сохраняем файл
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"[COPY] Файл успешно сохранен")
            print(f"[COPY] Размер файла: {os.path.getsize(file_path)} байт")
            return True
        else:
            print(f"[COPY] Ошибка при копировании файла: HTTP {response.status_code}")
            print(f"[COPY] Текст ответа: {response.text}")
            return False
            
    except Exception as e:
        print(f"[COPY] Ошибка при копировании файла: {str(e)}")
        print(f"[COPY] Тип ошибки: {type(e).__name__}")
        return False 