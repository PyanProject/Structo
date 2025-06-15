#!/usr/bin/env python3
"""
Прокси-сервер для перенаправления запросов на локальный ПК
"""

import requests
import os
import shutil
from flask import Flask, request, Response, jsonify, send_from_directory
import logging
import sys
from config import LOCAL_PC_IP, LOCAL_PC_PORT

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("proxy_server.log"),
                       logging.StreamHandler(sys.stdout)
                   ])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Базовый URL локального сервера
LOCAL_SERVER_URL = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}"

# Директория для сохранения моделей на VDS
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "models")

# Создаем директорию, если она не существует
try:
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Директория для моделей создана или уже существует: {MODELS_DIR}")
    logger.info(f"Путь существует: {os.path.exists(MODELS_DIR)}")
    
    # Проверяем права доступа к директории
    if os.access(MODELS_DIR, os.W_OK):
        logger.info(f"Есть права на запись в директорию {MODELS_DIR}")
    else:
        logger.error(f"НЕТ прав на запись в директорию {MODELS_DIR}")
except Exception as e:
    logger.error(f"Ошибка при создании директории для моделей: {str(e)}")

@app.route('/proxy/health', methods=['GET'])
def health_check():
    """Проверка доступности локального сервера"""
    try:
        logger.info(f"Проксирование GET запроса на {LOCAL_SERVER_URL}/health")
        response = requests.get(f"{LOCAL_SERVER_URL}/health", timeout=10)
        logger.info(f"Получен ответ от локального сервера: {response.status_code}")
        
        # Возвращаем ответ от локального сервера
        return Response(
            response.content,
            status=response.status_code,
            content_type=response.headers.get('Content-Type', 'application/json')
        )
    except Exception as e:
        logger.error(f"Ошибка при проксировании запроса: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка при подключении к локальному серверу: {str(e)}"
        }), 500

@app.route('/proxy/status', methods=['GET'])
def server_status():
    """Проверка статуса локального сервера генерации"""
    try:
        logger.info(f"Проверка статуса локального сервера: {LOCAL_SERVER_URL}/health")
        logger.info(f"IP адрес: {LOCAL_PC_IP}, порт: {LOCAL_PC_PORT}")
        
        try:
            response = requests.get(f"{LOCAL_SERVER_URL}/health", timeout=5)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"Локальный сервер доступен: {data}")
                    return jsonify({
                        "status": "online",
                        "gpu_available": True,
                        "message": "Локальный сервер генерации доступен"
                    })
                except:
                    logger.info("Получен ответ от локального сервера, но не в формате JSON")
                    return jsonify({
                        "status": "online",
                        "gpu_available": True,
                        "message": "Локальный сервер доступен"
                    })
            else:
                logger.error(f"Локальный сервер вернул ошибку: статус {response.status_code}")
                return jsonify({
                    "status": "error",
                    "gpu_available": False,
                    "message": f"Сервер вернул ошибку: {response.status_code}"
                })
        except requests.exceptions.ConnectionError as conn_error:
            error_message = str(conn_error)
            logger.error(f"Ошибка подключения к локальному серверу: {error_message}")
            
            # Извлекаем более конкретную информацию об ошибке
            if "Connection refused" in error_message:
                detailed_message = "Соединение отклонено. Убедитесь, что локальный сервер запущен и порт открыт."
            elif "Name or service not known" in error_message:
                detailed_message = "Не удается разрешить имя хоста. Проверьте настройки DNS или IP-адрес."
            elif "Network is unreachable" in error_message:
                detailed_message = "Сеть недоступна. Проверьте сетевое подключение."
            elif "timed out" in error_message.lower():
                detailed_message = "Время ожидания истекло. Сервер не отвечает."
            else:
                detailed_message = f"Ошибка соединения: {error_message}"
                
            logger.error(f"Детали подключения: IP={LOCAL_PC_IP}, Port={LOCAL_PC_PORT}")
            logger.error(f"Детализированное сообщение: {detailed_message}")
            
            return jsonify({
                "status": "offline",
                "gpu_available": False,
                "message": f"Локальный сервер недоступен: {detailed_message}",
                "error_details": error_message[:200] + "..." if len(error_message) > 200 else error_message,
                "server_url": LOCAL_SERVER_URL
            })
        except requests.exceptions.Timeout:
            logger.error("Таймаут при подключении к локальному серверу")
            return jsonify({
                "status": "offline",
                "gpu_available": False,
                "message": "Локальный сервер не отвечает (таймаут)"
            })
        except Exception as e:
            error_message = str(e)
            logger.error(f"Другая ошибка при проверке локального сервера: {error_message}")
            return jsonify({
                "status": "offline",
                "gpu_available": False,
                "message": f"Ошибка подключения к локальному серверу: {error_message[:200]}",
                "error_details": error_message
            })
    except Exception as e:
        error_message = str(e)
        logger.error(f"Непредвиденная ошибка при проверке статуса сервера: {error_message}")
        return jsonify({
            "status": "error",
            "gpu_available": False,
            "message": f"Непредвиденная ошибка: {error_message[:200]}",
            "error_details": error_message
        })

@app.route('/proxy/generate', methods=['POST'])
def generate():
    """Проксирование запроса на генерацию модели"""
    try:
        logger.info(f"Проксирование POST запроса на {LOCAL_SERVER_URL}/generate")
        logger.info(f"Тело запроса: {request.get_data(as_text=True)}")
        
        # Получаем данные запроса
        data = request.get_json()
        
        # Отправляем запрос на локальный сервер
        response = requests.post(
            f"{LOCAL_SERVER_URL}/generate",
            json=data,
            timeout=300  # 5 минут таймаут
        )
        
        logger.info(f"Получен ответ от локального сервера: {response.status_code}")
        
        # Пытаемся получить JSON ответ
        try:
            response_data = response.json()
            logger.info(f"Ответ сервера: {response_data}")
            
            # Добавляем model_url, если есть filename, но нет model_url
            if 'filename' in response_data and 'model_url' not in response_data and 'obj_url' not in response_data:
                filename = response_data['filename']
                # Запоминаем filename для последующего скачивания
                response_data['filename'] = filename
                
                # Формируем URL для модели на VDS
                vds_model_path = f"/static/models/{filename}"
                response_data['model_url'] = vds_model_path
                logger.info(f"Добавлен model_url для VDS: {response_data['model_url']}")
                
                # Формируем URL для скачивания с VDS
                response_data['download_url'] = f"/static/models/{filename}?download=true"
                logger.info(f"Добавлен download_url для VDS: {response_data['download_url']}")
                
                # Возвращаем модифицированный JSON
                return jsonify(response_data)
            
        except Exception as e:
            logger.warning(f"Не удалось прочитать JSON из ответа: {str(e)}")
        
        # Если не удалось обработать ответ как JSON, возвращаем оригинальный ответ
        return Response(
            response.content,
            status=response.status_code,
            content_type=response.headers.get('Content-Type', 'application/json')
        )
    except Exception as e:
        logger.error(f"Ошибка при проксировании запроса: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка при подключении к локальному серверу: {str(e)}"
        }), 500

@app.route('/proxy/progress', methods=['GET'])
def progress():
    """Проксирование запроса на получение прогресса генерации"""
    try:
        logger.info(f"Проксирование GET запроса на {LOCAL_SERVER_URL}/progress")
        
        # Отправляем запрос на локальный сервер
        response = requests.get(f"{LOCAL_SERVER_URL}/progress", timeout=10)
        logger.info(f"Получен ответ от локального сервера: {response.status_code}")
        
        # Получаем данные о прогрессе
        try:
            progress_data = response.json()
            logger.info(f"Данные о прогрессе: {progress_data}")
            
            # Если генерация завершена, скачиваем файл на VDS
            if progress_data.get('status') == 'completed' and progress_data.get('filename'):
                filename = progress_data.get('filename')
                logger.info(f"Генерация завершена, filename: {filename}")
                
                local_file_url = f"{LOCAL_SERVER_URL}/output/{filename}"
                vds_file_path = os.path.join(MODELS_DIR, filename)
                
                logger.info(f"URL локального файла: {local_file_url}")
                logger.info(f"Путь к файлу на VDS: {vds_file_path}")
                
                # Проверяем, скачан ли уже файл
                if not os.path.exists(vds_file_path):
                    logger.info(f"Файл не найден на VDS, скачиваем: {local_file_url}")
                    
                    # Скачиваем файл модели с локального ПК
                    try:
                        model_response = requests.get(local_file_url, stream=True, timeout=60)
                        logger.info(f"Ответ при скачивании файла: {model_response.status_code}")
                        
                        if model_response.status_code == 200:
                            # Проверяем, что файл не пустой
                            content_length = int(model_response.headers.get('Content-Length', '0'))
                            logger.info(f"Размер файла: {content_length} байт")
                            
                            if content_length > 0:
                                # Создаем временный файл для проверки целостности
                                temp_file_path = vds_file_path + ".tmp"
                                
                                try:
                                    with open(temp_file_path, 'wb') as f:
                                        for chunk in model_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                            
                                    # Проверяем размер скачанного файла
                                    file_size = os.path.getsize(temp_file_path)
                                    logger.info(f"Скачанный файл: {file_size} байт")
                                    
                                    if file_size > 0 and file_size == content_length:
                                        # Переименовываем временный файл в окончательный
                                        os.rename(temp_file_path, vds_file_path)
                                        logger.info(f"Файл успешно сохранен на VDS: {vds_file_path}")
                                        
                                        # Проверяем наличие файла и его размер после переименования
                                        if os.path.exists(vds_file_path):
                                            final_size = os.path.getsize(vds_file_path)
                                            logger.info(f"Проверка файла после переименования: {vds_file_path} ({final_size} байт)")
                                            
                                            # Добавляем информацию о пути к файлу на VDS
                                            progress_data['vds_model_url'] = f"/static/models/{filename}"
                                            progress_data['vds_download_url'] = f"/static/models/{filename}?download=true"
                                        else:
                                            logger.error(f"Файл не найден после переименования: {vds_file_path}")
                                    else:
                                        logger.error(f"Размер скачанного файла не соответствует ожидаемому: {file_size} != {content_length}")
                                        os.remove(temp_file_path)
                                except Exception as file_error:
                                    logger.error(f"Ошибка при сохранении файла: {str(file_error)}")
                                    if os.path.exists(temp_file_path):
                                        os.remove(temp_file_path)
                            else:
                                logger.error(f"Скачиваемый файл пустой (размер: {content_length})")
                        else:
                            logger.error(f"Не удалось скачать файл: статус {model_response.status_code}")
                    except Exception as e:
                        logger.error(f"Ошибка при скачивании модели: {str(e)}")
                else:
                    # Файл уже существует на VDS
                    file_size = os.path.getsize(vds_file_path)
                    logger.info(f"Файл уже существует на VDS: {vds_file_path} ({file_size} байт)")
                    progress_data['vds_model_url'] = f"/static/models/{filename}"
                    progress_data['vds_download_url'] = f"/static/models/{filename}?download=true"
            
            # Возвращаем обновленные данные о прогрессе
            return jsonify(progress_data)
            
        except Exception as e:
            logger.warning(f"Не удалось обработать ответ о прогрессе: {str(e)}")
        
        # Возвращаем ответ от локального сервера
        return Response(
            response.content,
            status=response.status_code,
            content_type=response.headers.get('Content-Type', 'application/json')
        )
    except Exception as e:
        logger.error(f"Ошибка при проксировании запроса: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка при подключении к локальному серверу: {str(e)}"
        }), 500

@app.route('/proxy/download/<path:filename>', methods=['GET'])
def download_model(filename):
    """Скачивание модели с локального сервера на VDS"""
    try:
        logger.info(f"Запрос на скачивание модели: {filename}")
        local_file_url = f"{LOCAL_SERVER_URL}/output/{filename}"
        vds_file_path = os.path.join(MODELS_DIR, filename)
        
        logger.info(f"URL локального файла: {local_file_url}")
        logger.info(f"Путь к файлу на VDS: {vds_file_path}")
        logger.info(f"Проверка подключения к локальному серверу: {LOCAL_SERVER_URL}")
        
        # Проверяем существование директории
        if not os.path.exists(MODELS_DIR):
            try:
                os.makedirs(MODELS_DIR, exist_ok=True)
                logger.info(f"Создана директория для моделей: {MODELS_DIR}")
            except Exception as dir_error:
                logger.error(f"Не удалось создать директорию: {str(dir_error)}")
                return jsonify({
                    "status": "error",
                    "message": f"Не удалось создать директорию для моделей: {str(dir_error)}"
                }), 500
        
        # Проверяем, существует ли файл на VDS
        if os.path.exists(vds_file_path):
            file_size = os.path.getsize(vds_file_path)
            logger.info(f"Файл найден на VDS: {vds_file_path} ({file_size} байт)")
            
            # Проверяем, что файл не пустой
            if file_size > 0:
                return jsonify({
                    "status": "success", 
                    "message": "Файл уже скачан на VDS",
                    "vds_model_url": f"/static/models/{filename}",
                    "vds_download_url": f"/static/models/{filename}?download=true",
                    "file_size": file_size
                })
            else:
                logger.warning(f"Файл существует, но имеет нулевой размер: {vds_file_path}")
                # Удаляем пустой файл
                os.remove(vds_file_path)
                logger.info(f"Удален пустой файл: {vds_file_path}")
        
        # Сначала проверим доступность локального сервера напрямую
        try:
            # Проверяем доступность сервера через запрос к /health
            logger.info(f"Проверка доступности локального сервера через /health")
            health_check = requests.get(f"{LOCAL_SERVER_URL}/health", timeout=5)
            logger.info(f"Локальный сервер доступен: статус {health_check.status_code}")
        except requests.exceptions.ConnectionError as conn_error:
            error_message = str(conn_error)
            logger.error(f"Сервер недоступен. Ошибка подключения: {error_message}")
            # Извлекаем более конкретную информацию об ошибке
            if "Connection refused" in error_message:
                detailed_message = "Соединение отклонено. Убедитесь, что локальный сервер запущен и порт открыт."
            elif "Name or service not known" in error_message:
                detailed_message = "Не удается разрешить имя хоста. Проверьте настройки DNS или IP-адрес."
            elif "Network is unreachable" in error_message:
                detailed_message = "Сеть недоступна. Проверьте сетевое подключение."
            elif "timed out" in error_message.lower():
                detailed_message = "Время ожидания истекло. Сервер не отвечает."
            else:
                detailed_message = f"Ошибка соединения: {error_message}"
            
            return jsonify({
                "status": "error",
                "message": f"Локальный сервер недоступен: {detailed_message}",
                "error_details": error_message
            }), 503
        except Exception as health_error:
            logger.error(f"Ошибка при проверке доступности локального сервера: {str(health_error)}")
            return jsonify({
                "status": "error",
                "message": f"Ошибка при проверке доступности локального сервера: {str(health_error)}"
            }), 500
        
        # Проверяем доступность файла на локальном сервере
        try:
            # Сначала проверяем HEAD-запросом, доступен ли файл
            logger.info(f"Проверка доступности файла с помощью HEAD-запроса: {local_file_url}")
            head_response = requests.head(local_file_url, timeout=10)
            if head_response.status_code != 200:
                logger.error(f"Файл недоступен на локальном сервере: {local_file_url} (статус: {head_response.status_code})")
                return jsonify({
                    "status": "error",
                    "message": f"Файл недоступен на локальном сервере: статус {head_response.status_code}"
                }), 404
            
            # Получаем размер файла
            content_length = int(head_response.headers.get('Content-Length', '0'))
            logger.info(f"Размер файла на локальном сервере: {content_length} байт")
            
            if content_length == 0:
                logger.error(f"Файл на локальном сервере имеет нулевой размер: {local_file_url}")
                return jsonify({
                    "status": "error",
                    "message": "Файл на локальном сервере имеет нулевой размер"
                }), 400
        except requests.exceptions.ConnectionError as conn_error:
            error_message = str(conn_error)
            logger.error(f"Ошибка подключения при проверке файла: {error_message}")
            
            # Извлекаем более конкретную информацию об ошибке
            if "Connection refused" in error_message:
                detailed_message = "Соединение отклонено. Убедитесь, что локальный сервер запущен и порт открыт."
            elif "Name or service not known" in error_message:
                detailed_message = "Не удается разрешить имя хоста. Проверьте настройки DNS или IP-адрес."
            elif "Network is unreachable" in error_message:
                detailed_message = "Сеть недоступна. Проверьте сетевое подключение."
            elif "timed out" in error_message.lower():
                detailed_message = "Время ожидания истекло. Сервер не отвечает."
            else:
                detailed_message = f"Ошибка соединения: {error_message}"
            
            # Логируем IP и порт для диагностики
            logger.error(f"Детали подключения: IP={LOCAL_PC_IP}, Port={LOCAL_PC_PORT}")
            
            return jsonify({
                "status": "error",
                "message": f"Ошибка при проверке файла на локальном сервере: {detailed_message}",
                "error_details": error_message[:200] + "..." if len(error_message) > 200 else error_message
            }), 503
        except Exception as head_error:
            error_message = str(head_error)
            logger.error(f"Ошибка при проверке файла на локальном сервере: {error_message}")
            return jsonify({
                "status": "error",
                "message": f"Ошибка при проверке файла на локальном сервере: {error_message[:200]}",
                "error_details": error_message
            }), 500
        
        # Скачиваем файл с локального сервера
        logger.info(f"Скачивание файла с локального ПК: {local_file_url}")
        temp_file_path = vds_file_path + ".tmp"
        
        try:
            response = requests.get(local_file_url, stream=True, timeout=60)
            
            if response.status_code == 200:
                # Сохраняем во временный файл
                bytes_downloaded = 0
                with open(temp_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                
                # Проверяем размер файла
                file_size = os.path.getsize(temp_file_path)
                logger.info(f"Скачано {bytes_downloaded} байт, размер файла: {file_size} байт")
                
                if file_size > 0:
                    # Переименовываем временный файл
                    os.rename(temp_file_path, vds_file_path)
                    logger.info(f"Файл успешно сохранен на VDS: {vds_file_path}")
                    
                    # Проверяем наличие файла после переименования
                    if os.path.exists(vds_file_path):
                        final_size = os.path.getsize(vds_file_path)
                        logger.info(f"Проверка файла после переименования: {vds_file_path} ({final_size} байт)")
                        
                        return jsonify({
                            "status": "success", 
                            "message": "Файл успешно скачан на VDS",
                            "vds_model_url": f"/static/models/{filename}",
                            "vds_download_url": f"/static/models/{filename}?download=true",
                            "file_size": final_size
                        })
                    else:
                        logger.error(f"Файл не найден после переименования: {vds_file_path}")
                        return jsonify({
                            "status": "error",
                            "message": "Ошибка после переименования временного файла"
                        }), 500
                else:
                    logger.error(f"Скачанный файл имеет нулевой размер")
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    return jsonify({
                        "status": "error",
                        "message": "Скачанный файл имеет нулевой размер"
                    }), 400
            else:
                logger.error(f"Не удалось скачать файл: статус {response.status_code}")
                return jsonify({
                    "status": "error", 
                    "message": f"Не удалось скачать файл: статус {response.status_code}"
                }), response.status_code
        except requests.exceptions.ConnectionError as conn_error:
            error_message = str(conn_error)
            logger.error(f"Ошибка подключения при скачивании файла: {error_message}")
            
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
            # Извлекаем более конкретную информацию об ошибке
            if "Connection refused" in error_message:
                detailed_message = "Соединение отклонено. Убедитесь, что локальный сервер запущен и порт открыт."
            elif "Name or service not known" in error_message:
                detailed_message = "Не удается разрешить имя хоста. Проверьте настройки DNS или IP-адрес."
            elif "Network is unreachable" in error_message:
                detailed_message = "Сеть недоступна. Проверьте сетевое подключение."
            elif "timed out" in error_message.lower():
                detailed_message = "Время ожидания истекло. Сервер не отвечает."
            else:
                detailed_message = f"Ошибка соединения: {error_message}"
                
            return jsonify({
                "status": "error",
                "message": f"Ошибка при скачивании файла: {detailed_message}",
                "error_details": error_message[:200] + "..." if len(error_message) > 200 else error_message
            }), 503
        except Exception as download_error:
            error_message = str(download_error)
            logger.error(f"Ошибка при скачивании файла: {error_message}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({
                "status": "error",
                "message": f"Ошибка при скачивании файла: {error_message[:200]}",
                "error_details": error_message
            }), 500
            
    except Exception as e:
        error_message = str(e)
        logger.error(f"Ошибка при скачивании модели: {error_message}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка при скачивании модели: {error_message[:200]}",
            "error_details": error_message
        }), 500

@app.route('/proxy/serve_model/<path:filename>', methods=['GET', 'HEAD', 'OPTIONS'])
def serve_model_proxy(filename):
    """Отдача файла модели напрямую через прокси-сервер, обходя NGINX"""
    # Обработка OPTIONS запросов для CORS
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        return response
    
    try:
        file_path = os.path.join(MODELS_DIR, filename)
        logger.info(f"Запрос {request.method} на получение файла через прокси: {filename}")
        logger.info(f"Полный путь к файлу: {file_path}")
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            logger.error(f"ОШИБКА: Файл не найден через прокси: {file_path}")
            
            # Проверяем содержимое директории для диагностики
            logger.info(f"Содержимое директории {MODELS_DIR}:")
            try:
                files = os.listdir(MODELS_DIR)
                for f in files:
                    file_size = os.path.getsize(os.path.join(MODELS_DIR, f))
                    logger.info(f"  - {f} ({file_size} байт)")
                
                if not files:
                    logger.info("  Директория пуста")
            except Exception as dir_error:
                logger.error(f"Ошибка при чтении директории: {str(dir_error)}")
            
            # Для HEAD-запросов просто возвращаем 404 с заголовками
            if request.method == 'HEAD':
                response = Response()
                response.status_code = 404
                response.headers.add("Access-Control-Allow-Origin", "*")
                response.headers.add("Access-Control-Allow-Headers", "Content-Type")
                response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                return response
            
            # Попробуем скачать файл с локального сервера еще раз (только для GET-запросов)
            if request.method == 'GET':
                try:
                    local_file_url = f"{LOCAL_SERVER_URL}/output/{filename}"
                    logger.info(f"Попытка скачать файл с локального сервера: {local_file_url}")
                    
                    response = requests.get(local_file_url, stream=True, timeout=60)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        logger.info(f"Файл успешно загружен с локального сервера: {file_path}")
                        
                        # Проверим, что файл существует и имеет размер
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            logger.info(f"Файл готов к отправке: {file_path} ({os.path.getsize(file_path)} байт)")
                            
                            # Используем собственную функцию для отправки файла с правильными заголовками
                            return serve_file_with_headers(file_path, filename)
                    else:
                        logger.error(f"Не удалось загрузить файл с локального сервера: статус {response.status_code}")
                except Exception as download_error:
                    logger.error(f"Ошибка при попытке загрузить файл с локального сервера: {str(download_error)}")
            
            # Файл не найден и не удалось его восстановить
            response = jsonify({
                "error": "File not found",
                "message": f"Файл {filename} не найден на сервере"
            })
            
            # Добавляем CORS-заголовки
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            
            return response, 404
        
        # Если файл существует
        file_size = os.path.getsize(file_path)
        logger.info(f"Отдача файла через прокси: {filename} ({file_size} байт)")
        
        # Определяем MIME-тип
        mime_type = "application/octet-stream"  # По умолчанию
        if filename.lower().endswith('.obj'):
            mime_type = "text/plain"  # OBJ файлы - это текстовый формат
        elif filename.lower().endswith('.mtl'):
            mime_type = "text/plain"  # MTL файлы - тоже текстовый формат
        elif filename.lower().endswith('.fbx'):
            mime_type = "application/octet-stream"
        elif filename.lower().endswith('.glb') or filename.lower().endswith('.gltf'):
            mime_type = "model/gltf-binary"
        
        # Для HEAD-запросов возвращаем только заголовки
        if request.method == 'HEAD':
            response = Response()
            response.headers.add("Content-Type", mime_type)
            response.headers.add("Content-Length", str(file_size))
            response.headers.add("Content-Disposition", f"inline; filename={filename}")
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            response.headers.add("Cache-Control", "no-cache, no-store, must-revalidate")
            return response
        
        # Для GET-запросов отправляем файл с заголовками
        return serve_file_with_headers(file_path, filename)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса к файлу через прокси {filename}: {str(e)}")
        response = jsonify({
            "error": "Internal server error",
            "message": f"Ошибка при обработке запроса: {str(e)}"
        })
        
        # Добавляем CORS-заголовки
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        
        return response, 500

# Вспомогательная функция для отправки файла с правильными заголовками
def serve_file_with_headers(file_path, filename):
    """Отправляет файл с правильными MIME-типами и CORS-заголовками"""
    try:
        # Определяем MIME-тип в зависимости от расширения файла
        mime_type = "application/octet-stream"  # По умолчанию
        extension = os.path.splitext(filename.lower())[1]
        
        mime_types = {
            '.obj': 'text/plain',
            '.mtl': 'text/plain',
            '.fbx': 'application/octet-stream',
            '.glb': 'model/gltf-binary',
            '.gltf': 'model/gltf+json',
            '.bin': 'application/octet-stream',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tga': 'image/x-tga',
            '.dds': 'image/vnd.ms-dds',
            '.hdr': 'image/vnd.radiance',
            '.exr': 'image/x-exr',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.svg': 'image/svg+xml',
            '.babylon': 'application/babylon',
            '.stl': 'model/stl',
            '.ply': 'application/octet-stream',
            '.vrm': 'model/vrm',
            '.usdz': 'model/vnd.usdz+zip',
            '.usda': 'model/vnd.usda',
            '.usdc': 'model/vnd.usdc',
            '.usd': 'model/vnd.usd'
        }
        
        mime_type = mime_types.get(extension, mime_type)
        
        # Читаем содержимое файла
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Отправка файла {filename} ({file_size} байт) с MIME-типом {mime_type}")
        
        # Создаем ответ
        response = Response(file_data, mimetype=mime_type)
        
        # Добавляем необходимые заголовки
        response.headers.add("Content-Disposition", f"inline; filename={filename}")
        response.headers.add("Content-Length", str(file_size))
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Range, Accept, Accept-Encoding, Content-Length")
        response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        response.headers.add("Access-Control-Expose-Headers", "Content-Length, Content-Range, Content-Encoding")
        
        # Заголовки кеширования - для лучшей производительности разрешаем кеширование на 1 час
        response.headers.add("Cache-Control", "public, max-age=3600")
        
        # Добавляем ETag для условной загрузки
        import hashlib
        etag = hashlib.md5(file_data[:1024] + str(file_size).encode()).hexdigest()
        response.headers.add("ETag", f'"{etag}"')
        
        return response
    except Exception as e:
        logger.error(f"Ошибка при отправке файла {filename} с заголовками: {str(e)}")
        # Если произошла ошибка, пробуем обычный метод
        return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path))

@app.route('/static/models/<path:filename>', methods=['GET', 'HEAD', 'OPTIONS'])
def serve_model(filename):
    """Отдача файла модели с VDS"""
    # Обработка OPTIONS запросов для CORS
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Range, Accept, Accept-Encoding, Content-Length")
        response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        response.headers.add("Access-Control-Expose-Headers", "Content-Length, Content-Range, Content-Encoding")
        return response
    
    try:
        file_path = os.path.join(MODELS_DIR, filename)
        logger.info(f"Запрос {request.method} на получение файла: {filename}")
        logger.info(f"Полный путь к файлу: {file_path}")
        logger.info(f"Файл существует: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            logger.error(f"ОШИБКА: Файл не найден: {file_path}")
            
            # Проверяем содержимое директории
            logger.info(f"Содержимое директории {MODELS_DIR}:")
            try:
                files = os.listdir(MODELS_DIR)
                for f in files:
                    file_size = os.path.getsize(os.path.join(MODELS_DIR, f))
                    logger.info(f"  - {f} ({file_size} байт)")
                
                if not files:
                    logger.info("  Директория пуста")
            except Exception as dir_error:
                logger.error(f"Ошибка при чтении директории: {str(dir_error)}")
            
            # Для HEAD-запросов просто возвращаем 404 с заголовками
            if request.method == 'HEAD':
                response = Response()
                response.status_code = 404
                response.headers.add("Access-Control-Allow-Origin", "*")
                response.headers.add("Access-Control-Allow-Headers", "Content-Type, Range, Accept, Accept-Encoding, Content-Length")
                response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                return response
            
            response = jsonify({
                "error": "File not found",
                "message": f"Файл {filename} не найден на сервере"
            })
            
            # Добавляем CORS-заголовки
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            
            return response, 404
        
        download = request.args.get('download', 'false').lower() == 'true'
        logger.info(f"Параметр download: {download}")
        file_size = os.path.getsize(file_path)
        logger.info(f"Размер файла: {file_size} байт")
        
        # Для HEAD-запросов не скачиваем файл, только отправляем заголовки
        if request.method == 'HEAD':
            extension = os.path.splitext(filename.lower())[1]
            mime_type = "application/octet-stream"  # По умолчанию
            
            # Определяем MIME-тип в зависимости от расширения
            if extension in ['.obj', '.mtl']:
                mime_type = "text/plain"
            elif extension in ['.glb']:
                mime_type = "model/gltf-binary"
            elif extension in ['.gltf']:
                mime_type = "model/gltf+json"
            
            response = Response()
            response.headers.add("Content-Type", mime_type)
            response.headers.add("Content-Length", str(file_size))
            response.headers.add("Content-Disposition", f"inline; filename={filename}")
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type, Range, Accept, Accept-Encoding, Content-Length")
            response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            response.headers.add("Cache-Control", "public, max-age=3600")
            return response
        
        if download:
            logger.info(f"Отдача файла для скачивания: {filename}")
            response = send_from_directory(
                MODELS_DIR, 
                filename, 
                as_attachment=True,
                download_name=filename
            )
            # Добавим CORS-заголовки
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type, Range, Accept, Accept-Encoding, Content-Length")
            response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            return response
        else:
            logger.info(f"Отдача файла для просмотра: {filename}")
            # Используем нашу функцию с заголовками
            return serve_file_with_headers(file_path, filename)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса к файлу {filename}: {str(e)}")
        response = jsonify({
            "error": "Internal server error",
            "message": f"Ошибка при обработке запроса: {str(e)}"
        })
        
        # Добавляем CORS-заголовки
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        
        return response, 500

@app.route('/proxy/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    """Общий метод для проксирования любых запросов"""
    try:
        url = f"{LOCAL_SERVER_URL}/{path}"
        logger.info(f"Проксирование {request.method} запроса на {url}")
        
        # Пересылаем запрос на локальный сервер
        response = requests.request(
            method=request.method,
            url=url,
            headers={key: value for key, value in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=300
        )
        
        logger.info(f"Получен ответ от локального сервера: {response.status_code}")
        
        # Возвращаем ответ от локального сервера
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers)
        )
    except Exception as e:
        logger.error(f"Ошибка при проксировании запроса: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ошибка при подключении к локальному серверу: {str(e)}"
        }), 500

@app.route('/proxy/diagnose', methods=['GET'])
def diagnose_connection():
    """Диагностика подключения к локальному серверу"""
    # Можно передать другие IP и порт для проверки
    test_ip = request.args.get('ip', LOCAL_PC_IP)
    test_port = request.args.get('port', LOCAL_PC_PORT)
    
    logger.info(f"Запуск диагностики подключения к серверу: IP={test_ip}, Port={test_port}")
    
    # Информация о сетевом окружении
    import socket
    import subprocess
    
    results = {
        "server_config": {
            "default_ip": LOCAL_PC_IP,
            "default_port": LOCAL_PC_PORT,
            "test_ip": test_ip,
            "test_port": test_port,
            "server_url": f"http://{test_ip}:{test_port}"
        },
        "network_info": {},
        "connection_tests": {}
    }
    
    # Получаем имя хоста и IP прокси-сервера
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        results["network_info"]["hostname"] = hostname
        results["network_info"]["proxy_ip"] = local_ip
    except Exception as e:
        results["network_info"]["error"] = f"Не удалось получить сетевую информацию: {str(e)}"
    
    # Пинг до тестируемого IP
    try:
        ping_cmd = f"ping -c 4 {test_ip}"
        ping_result = subprocess.run(ping_cmd, shell=True, capture_output=True, text=True)
        if ping_result.returncode == 0:
            results["connection_tests"]["ping"] = {
                "status": "success",
                "output": ping_result.stdout
            }
        else:
            results["connection_tests"]["ping"] = {
                "status": "failed",
                "output": ping_result.stderr or ping_result.stdout
            }
    except Exception as e:
        results["connection_tests"]["ping"] = {
            "status": "error",
            "message": f"Ошибка при выполнении ping: {str(e)}"
        }
    
    # Попытка соединения через сокет
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((test_ip, int(test_port)))
        if result == 0:
            results["connection_tests"]["socket"] = {
                "status": "success",
                "message": "Порт открыт"
            }
        else:
            results["connection_tests"]["socket"] = {
                "status": "failed",
                "message": f"Порт закрыт или недоступен (код ошибки: {result})"
            }
        sock.close()
    except Exception as e:
        results["connection_tests"]["socket"] = {
            "status": "error",
            "message": f"Ошибка при проверке сокета: {str(e)}"
        }
    
    # HTTP-запрос к серверу
    test_url = f"http://{test_ip}:{test_port}/health"
    try:
        http_response = requests.get(test_url, timeout=5)
        results["connection_tests"]["http"] = {
            "status": "success",
            "status_code": http_response.status_code,
            "content_type": http_response.headers.get('Content-Type', 'неизвестно'),
            "response_time": http_response.elapsed.total_seconds(),
            "response_size": len(http_response.content)
        }
        
        try:
            results["connection_tests"]["http"]["json_response"] = http_response.json()
        except:
            if len(http_response.content) < 500:
                results["connection_tests"]["http"]["text_response"] = http_response.text
    except requests.exceptions.ConnectionError as e:
        results["connection_tests"]["http"] = {
            "status": "failed",
            "error_type": "ConnectionError",
            "message": str(e)
        }
    except requests.exceptions.Timeout as e:
        results["connection_tests"]["http"] = {
            "status": "failed",
            "error_type": "Timeout",
            "message": "Время ожидания истекло"
        }
    except Exception as e:
        results["connection_tests"]["http"] = {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }
    
    # Общий результат диагностики
    if results["connection_tests"].get("http", {}).get("status") == "success":
        results["status"] = "online"
        results["message"] = "Сервер доступен"
    elif results["connection_tests"].get("socket", {}).get("status") == "success":
        results["status"] = "partially_available"
        results["message"] = "Порт открыт, но HTTP-сервис недоступен"
    elif results["connection_tests"].get("ping", {}).get("status") == "success":
        results["status"] = "host_available"
        results["message"] = "Хост доступен, но порт закрыт"
    else:
        results["status"] = "offline"
        results["message"] = "Сервер полностью недоступен"
    
    logger.info(f"Результат диагностики: {results['status']} - {results['message']}")
    
    return jsonify(results)

if __name__ == '__main__':
    logger.info(f"Запуск прокси-сервера на порту 5002")
    logger.info(f"Локальный сервер: {LOCAL_SERVER_URL}")
    logger.info(f"Директория для моделей на VDS: {MODELS_DIR}")
    logger.info(f"Проверка существования директории: {os.path.exists(MODELS_DIR)}")
    app.run(host='0.0.0.0', port=5002, debug=True) 