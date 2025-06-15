from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, session, send_file, Response, make_response
import os
import sqlite3
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tempfile
from werkzeug.utils import secure_filename
import base64
import requests
from config import GENERATION_API_URL, HEALTH_CHECK_URL, REQUEST_TIMEOUT
from remote_generator import check_generator_status, generate_model, serve_model_file, download_model_file, check_generation_progress
from flask_cors import CORS
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from random import randint
import trimesh
import numpy as np
from io import BytesIO

# Функция для нормализации путей (замена обратных слешей на прямые)
def normalize_path(path):
    """
    Заменяет обратные слеши на прямые в пути файла.
    
    Args:
        path (str): Путь к файлу
        
    Returns:
        str: Нормализованный путь с прямыми слешами
    """
    if path and '\\' in path:
        print(f"[NORMALIZE] Обнаружены обратные слеши в пути: {path}")
        normalized = path.replace('\\', '/')
        print(f"[NORMALIZE] Нормализованный путь: {normalized}")
        return normalized
    return path

# =============================================================================
# Инициализация приложения и конфигурация
# =============================================================================

# Создаем приложение с относительной конфигурацией
app = Flask(__name__, instance_relative_config=True)
CORS(app)

# Задаем путь для загрузки сгенерированных моделей
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'models')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Создаем папку instance для базы данных, если её ещё нет
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.path.join(app.instance_path, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DB_PATH + '?mode=rw'

# Другие настройки приложения
app.config['SECRET_KEY'] = '8fe72d74a14c657fa1b81fc4ffdc4c6e8a13243f48a97110'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_DURATION'] = timedelta(days=30)

# =============================================================================
# Инициализация базы данных
# =============================================================================

db = SQLAlchemy(app)

class User(db.Model):
    """Модель пользователя."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    confirmed = db.Column(db.Integer, default=0)
    confirmation_code = db.Column(db.Integer, default=0)
    display_name = db.Column(db.String(150))
    subscription = db.Column(db.Integer, default=1)

class DownloadedFile(db.Model):
    """Модель загруженного файла."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(150), nullable=False)
    prompt = db.Column(db.String(500))  # Добавляем сохранение промпта
    download_time = db.Column(db.DateTime, default=datetime.utcnow)

def check_db_tables():
    """Проверяет структуру таблицы 'users' в базе данных."""
    if not os.path.exists(DB_PATH):
        print(f"[DB CHECK] Файл базы данных {DB_PATH} не найден.")
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info('users');")
    columns = cursor.fetchall()
    print("[DB CHECK] Структура таблицы 'users':", columns)
    conn.close()

check_db_tables()

# =============================================================================
# Основные страницы
# =============================================================================

@app.before_request
def before_request():
    # Полностью отключаем перенаправление на HTTPS в Flask
    # Nginx будет заниматься этим
    pass
    
    # Старый код, закомментированный:
    # if (not request.is_secure and 
    #     not request.path.startswith('/check_generator_status') and
    #     not request.path.startswith('/generate_model') and
    #     not request.path.startswith('/models/') and
    #     not request.path.startswith('/download_model/') and
    #     not request.host.startswith('127.0.0.1') and
    #     not request.host.startswith('localhost')):
    #     
    #     url = request.url.replace('http://', 'https://', 1)
    #     return redirect(url, code=301)

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prompt_guide')
def prompt_guide():
    return render_template('prompt_guide.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/generate')
def main_page():
    return render_template('main_page.html')

@app.route('/view_model')
def view_model():
    """Страница для просмотра сгенерированной модели"""
    prompt = request.args.get('prompt', 'Без описания')
    filename = request.args.get('filename', '')
    model_url = request.args.get('model_url', '')
    download_url = request.args.get('download_url', '')
    
    # Если нет URL модели, перенаправляем на главную
    if not model_url:
        return redirect(url_for('main_page'))
    
    # Форматируем время создания (текущее время для простоты)
    created_at = datetime.now().strftime("%d.%m.%Y, %H:%M")
    
    return render_template(
        'view_model.html',
        prompt=prompt,
        filename=filename,
        model_url=model_url,
        download_url=download_url,
        created_at=created_at
    )

# =============================================================================
# Маршруты для аутентификации
# =============================================================================

@app.route('/auth_status', methods=['GET'])
def auth_status():
    """Проверка статуса авторизации"""
    try:
        if 'user_id' not in session:
            return jsonify({'authenticated': False}), 200
            
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'authenticated': False}), 200
            
        return jsonify({
            'authenticated': True,
            'username': user.username
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке авторизации: {str(e)}")
        return jsonify({
            'authenticated': False,
            'error': str(e)
        }), 500

@app.route('/auth', methods=['POST'])
def auth():
    print("[DEBUG] Получен запрос на /auth")
    print("[DEBUG] Content-Type:", request.headers.get('Content-Type'))
    print("[DEBUG] Тело запроса:", request.get_data(as_text=True))
    
    try:
        data = request.get_json()
        action = data.get('action')
        print(f"[DEBUG] Action: {action}")

        if action == 'login':
            username = data.get('username')
            password = data.get('password')
            remember_me = data.get('remember_me', False)
            user = User.query.filter_by(username=username).first()
            check_password = check_password_hash(user.password, password)
            if user and check_password:
                session['user_id'] = user.id
                session.permanent = remember_me
                print(f"[DEBUG] Успешный вход: user_id={user.id}, remember_me={remember_me}, "
                      f"session.permanent={session.permanent}")
                return jsonify({'success': True})
            elif user and not check_password:
                return jsonify({'success': False, 'message': 'Неверный пароль'}), 401
            elif not user and check_password:
                return jsonify({'success': False, 'message': 'Неверный логин'}), 401
            else:
                return jsonify({'success': False, 'message': 'Неверный логин и пароль'}), 401

        elif action == 'register':
            username = data.get('username')
            password = data.get('password')
            email = data.get('email')
            confirmation_code = randint(100000, 999999)
            print(f"[DEBUG] Регистрация пользователя: {username}, {email}")
            
            if User.query.filter_by(username=username).first():
                response = jsonify({'success': False, 'message': 'Пользователь с таким логином уже существует'})
                print("[DEBUG] Ответ:", response.get_data(as_text=True))
                return response, 400

            password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
            new_user = User(username=username, password=password_hash, email=email, confirmation_code=confirmation_code)
            db.session.add(new_user)
            db.session.commit()
            db.session.close()
            db.session.begin()

            smtp_server = "smtp.yandex.ru"
            smtp_port = 465
            sender_email = "info.structo@yandex.ru"  
            password = "quauawansqwxrrly"

            try:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = email
                msg['Subject'] = "Добро пожаловать!"

                body = f"""
                Здравствуйте, {username}!
                Спасибо за регистрацию на проекте Structo.
                
                Для завершения регистрации введите код подтверждения на сайте: {confirmation_code}

                Если Вы не регистрировались на нашем сайте, проигнорируйте это письмо.
                
                С уважением,
                Команда поддержки Pyan Inc.
                """
                msg.attach(MIMEText(body, 'plain', 'utf-8'))  

                with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                    server.login(sender_email, password)
                    server.send_message(msg)
                
                print(f"[DEBUG] Письмо успешно отправлено на {email}")

            except Exception as e:
                print(f"[ERROR] Ошибка при отправке письма: {str(e)}")
                return jsonify({'success': False, 'message': str(e)}), 500
            
            response = jsonify({'success': True})
            print("[DEBUG] Ответ:", response.get_data(as_text=True))
            return response
        
        elif action == 'resend_code':
            email = data.get('email')
            confirmation_code = randint(100000, 999999)
            print(f"[DEBUG] Отправка кода подтверждения на {email}")
            
            smtp_server = "smtp.yandex.ru"
            smtp_port = 465
            sender_email = "info.structo@yandex.ru"  
            password = "quauawansqwxrrly"

            user = User.query.filter_by(email=email).first()

            user.confirmation_code = confirmation_code
            db.session.commit()

            try:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = email
                msg['Subject'] = "Повторный код подтверждения"

                body = f"""
                Высылаем Вам новый код подтверждения для регистрации на сайте: {confirmation_code}

                Если Вы не регистрировались на нашем сайте, проигнорируйте это письмо.
                
                С уважением,
                Команда поддержки Pyan Inc.
                """
                msg.attach(MIMEText(body, 'plain', 'utf-8'))  

                with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                    server.login(sender_email, password)
                    server.send_message(msg)
                
                print(f"[DEBUG] Письмо успешно отправлено на {email}")

            except Exception as e:
                print(f"[ERROR] Ошибка при отправке письма: {str(e)}")
                return jsonify({'success': False, 'message': str(e)}), 500
            
            response = jsonify({'success': True})
            print("[DEBUG] Ответ:", response.get_data(as_text=True))
            return response
            
    except Exception as e:
        print(f"[ERROR] Ошибка при обработке запроса: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get('lang', 'en')
    session['lang'] = lang
    return jsonify({'success': True})

def get_translation(key):
    translations = {
        'en': {
            'welcome': 'Welcome to your profile!',
            'logout_success': 'You have successfully logged out.',
        },
        'ru': {
            'welcome': 'Добро пожаловать в личный кабинет!',
            'logout_success': 'Вы успешно вышли из системы.',
        },
    }
    lang = session.get('lang', 'ru')
    return translations[lang].get(key, key)

# =============================================================================
# Маршруты для работы с генератором моделей
# =============================================================================

@app.route('/check_generator_status', methods=['GET'])
def check_remote_generator_status():
    """Проверка статуса генератора на удаленном ПК"""
    try:
        # Проверяем доступность сервера генерации
        import requests
        from config import HEALTH_CHECK_URL
        
        # Делаем запрос к API генератора
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        
        # Логгируем ответ для отладки
        print(f"Ответ от генератора: {response.status_code}, {response.text}")
        
        if response.status_code == 200:
            # Возвращаем настоящий статус генератора
            generator_data = response.json()
            # Гарантируем, что поле online существует
            generator_data['online'] = True
            return jsonify(generator_data)
        else:
            return jsonify({
                'status': 'error',
                'online': False,
                'message': f"Ошибка при проверке статуса генератора: HTTP {response.status_code}"
            }), response.status_code
    except Exception as e:
        # В случае ошибки (например, сервер недоступен)
        print(f"Ошибка при проверке генератора: {str(e)}")
        return jsonify({
            'status': 'error',
            'online': False,
            'message': f"Ошибка при проверке статуса генератора: {str(e)}"
        }), 503

@app.route('/generate_model', methods=['POST'])
def remote_generate_model():
    """Отправляет запрос на генерацию модели на удаленный сервер"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        # Проверка на пустой промпт
        if not prompt or prompt.strip() == '':
            return jsonify({
                'success': False,
                'error': 'Промпт не может быть пустым'
            })
        
        # Проверка статуса генератора
        # Вместо импортированной функции check_generator_status используем прямой запрос
        try:
            from config import HEALTH_CHECK_URL
            # Прямая проверка доступности генератора
            generator_response = requests.get(HEALTH_CHECK_URL, timeout=5)
            print(f"Прямой запрос к генератору: {generator_response.status_code}, {generator_response.text}")
        
            if generator_response.status_code == 200:
                generator_status = generator_response.json()
                # Явно устанавливаем флаг online в True
                generator_status['online'] = True
            else:
                return jsonify({
                    'success': False,
                    'error': 'Генератор моделей недоступен',
                    'status_details': {
                        'status': 'error',
                        'message': f'Ошибка при проверке статуса генератора: HTTP {generator_response.status_code}'
                    }
                })
        except Exception as health_check_error:
            print(f"Ошибка при проверке доступности генератора: {str(health_check_error)}")
            return jsonify({
                'success': False,
                'error': 'Не удалось подключиться к генератору моделей',
                'status_details': {
                    'status': 'error',
                    'message': str(health_check_error)
                }
            })
        
        # Определяем уровень подписки пользователя
        subscription_level = 1  # По умолчанию - базовый тариф
        
        # Если пользователь авторизован, получаем его тариф из базы данных
        if 'user_id' in session:
            try:
                user = User.query.get(session['user_id'])
                if user:
                    subscription_level = user.subscription
                    print(f"Пользователь {user.username} с тарифом {subscription_level}")
                else:
                    print(f"Пользователь с ID {session['user_id']} не найден")
            except Exception as e:
                print(f"Ошибка при получении информации о пользователе: {str(e)}")
        else:
            print("Неавторизованный пользователь, используется базовый тариф")
        
        # Отправка запроса на генерацию с учетом тарифа
        try:
            response = generate_model(prompt, subscription_level)
            print(f"Ответ от сервера генерации: {response}")
        
            # Проверка на ситуацию, когда генерация уже запущена (код 429)
            if response.get('is_generating', False) and response.get('status') == 'error':
                # Получаем информацию о текущем прогрессе генерации
                current_progress = response.get('current_progress', {})
                current_prompt = current_progress.get('prompt', 'Неизвестный промпт')
                progress_percent = current_progress.get('progress', 0)
                message = current_progress.get('message', '')
                
                return jsonify({
                    'success': False,
                    'error': 'Генерация уже запущена. Пожалуйста, дождитесь завершения текущего процесса.',
                    'is_generating': True,
                    'current_prompt': current_prompt,
                    'progress': progress_percent,
                    'message': message
                }), 429
            
            # Если в ответе есть ключ error, возвращаем ошибку
            if 'error' in response:
                return jsonify({
                    'success': False,
                    'error': response.get('error', 'Неизвестная ошибка при генерации модели'),
                    'status_details': response
                })
            
            # Проверка успешности выполнения запроса
            if response.get('status') != 'success' or response.get('success') == False:
                return jsonify({
                    'success': False,
                    'error': response.get('message', 'Ошибка при генерации модели'),
                    'status_details': response
                })
            
            # Сохраняем файл в историю, если пользователь авторизован
            filename = response.get('filename')
            if filename and 'user_id' in session:
                try:
                    # Создание записи о загруженном файле
                    new_file = DownloadedFile(
                        user_id=session['user_id'],
                        filename=filename,
                        prompt=prompt  # Сохраняем промпт вместе с файлом
                    )
                    db.session.add(new_file)
                    db.session.commit()
                    print(f"Файл {filename} добавлен в историю пользователя {session['user_id']}")
                except Exception as e:
                    print(f"Ошибка при сохранении файла в историю: {str(e)}")
            
            # Подготовка ответа
            result = {
                'success': True,
                'filename': filename,
                'prompt': prompt,
                'subscription_level': subscription_level
            }
            
            # Добавляем URL для просмотра модели
            if 'model_url' in response:
                result['model_url'] = response['model_url']
            
            # Добавляем URL для скачивания модели
            if 'download_url' in response:
                result['download_url'] = response['download_url']
            
            return jsonify(result)
            
        except Exception as generation_error:
            print(f"Ошибка при генерации модели: {str(generation_error)}")
            return jsonify({
                'success': False,
                'error': f'Ошибка при генерации модели: {str(generation_error)}'
            }), 500
        
    except Exception as e:
        print(f"Общая ошибка в запросе на генерацию: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Ошибка в запросе: {str(e)}'
        })

# Константы
MODELS_FOLDER = os.path.join(app.root_path, 'static', 'models')

# Создаем папку для моделей, если она не существует
os.makedirs(MODELS_FOLDER, exist_ok=True)

def copy_model_from_local_server(filename, prompt=None):
    """
    Копирует файл модели с локального сервера.
    
    Args:
        filename (str): Имя файла модели
        prompt (str, optional): Промпт, по которому генерировалась модель
        
    Returns:
        bool: True если копирование успешно, False в случае ошибки
    """
    try:
        from config import LOCAL_PC_IP, LOCAL_PC_PORT
        
        # Нормализуем имя файла (заменяем обратные слеши на прямые)
        filename = normalize_path(filename)
        
        print(f"[COPY] Начало копирования файла {filename} с локального сервера")
        print(f"[COPY] IP: {LOCAL_PC_IP}, порт: {LOCAL_PC_PORT}")
        
        # Проверяем соответствие промпта имени файла, если промпт указан
        if prompt:
            safe_prompt = ''.join(c if c.isalnum() else '_' for c in prompt.lower())
            if not filename.startswith(safe_prompt):
                print(f"[COPY] Ошибка: имя файла {filename} не соответствует промпту {prompt}")
                return False
        
        # Формируем URL для загрузки файла
        # Проверяем, начинается ли имя файла с 'output/' или нет
        if filename.startswith('output/'):
            filename = filename.replace('output/', '')
        
        url = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/output/{filename}"
        print(f"[COPY] URL для загрузки файла: {url}")
        
        # Отправляем GET запрос для получения файла
        print(f"[COPY] Отправка GET запроса с таймаутом 30 секунд...")
        response = requests.get(url, timeout=30)
        
        print(f"[COPY] Ответ получен. Статус: {response.status_code}")
        print(f"[COPY] Заголовки ответа: {dict(response.headers)}")
        print(f"[COPY] Размер ответа: {len(response.content)} байт")
        
        if response.status_code == 200:
            # Сохраняем файл в папку моделей
            file_path = os.path.join(MODELS_FOLDER, secure_filename(filename))
            print(f"[COPY] Сохранение файла в {file_path}")
            
            # Проверяем содержимое ответа
            if len(response.content) < 100:
                try:
                    print(f"[COPY] Содержимое ответа (первые 100 байт): {response.content[:100]}")
                except:
                    print(f"[COPY] Не удалось вывести содержимое ответа")
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Проверяем, что файл был успешно сохранен
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"[COPY] Файл успешно сохранен. Размер: {file_size} байт")
                
                # Проверка, что файл не пустой
                if file_size == 0:
                    print(f"[COPY] ВНИМАНИЕ: Сохраненный файл имеет нулевой размер!")
                    return False
                
            return True
        else:
                print(f"[COPY] Ошибка: файл не был сохранен")
                return False
        """else:
            print(f"[COPY] Ошибка при загрузке файла: HTTP {response.status_code}")
            if response.content:
                try:
                    print(f"[COPY] Ответ сервера: {response.text}")
                except:
                    print(f"[COPY] Не удалось вывести текст ответа")
            return False"""
            
    except requests.exceptions.ConnectionError as e:
        print(f"[COPY] Ошибка соединения: {str(e)}")
        print(f"[COPY] Проверьте, что ваш локальный сервер запущен и доступен по адресу {LOCAL_PC_IP}:{LOCAL_PC_PORT}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"[COPY] Таймаут запроса: {str(e)}")
        print(f"[COPY] Сервер не ответил в течение 30 секунд. Возможно, файл слишком большой или сервер перегружен.")
        return False
    except Exception as e:
        print(f"[COPY] Непредвиденная ошибка: {str(e)}")
        print(f"[COPY] Тип ошибки: {type(e).__name__}")
        import traceback
        print(f"[COPY] Трассировка: {traceback.format_exc()}")
        return False

@app.route('/models/<filename>')
def serve_model(filename):
    """Отдает файл модели для отображения на странице"""
    try:
        # Нормализуем имя файла, заменяя обратные слеши на прямые
        filename = normalize_path(filename)
        
        print(f"Запрос на получение модели: {filename}")
        
        # Безопасный путь к файлу
        safe_filename = secure_filename(filename)
        file_path = os.path.join(MODELS_FOLDER, safe_filename)
        print(f"Полный путь к файлу: {file_path}")
        
        # Если файл не существует или имеет нулевой размер, пытаемся скопировать его с локального сервера
        file_exists = os.path.exists(file_path)
        file_size = os.path.getsize(file_path) if file_exists else 0
        
        if not file_exists or file_size == 0:
            print(f"Файл модели не найден или имеет нулевой размер: {file_path}. Пытаемся скопировать с локального сервера.")
            
            # Пытаемся скопировать файл
            if copy_model_from_local_server(filename):
                print(f"Файл успешно скопирован с локального сервера")
                
                # Проверяем, что файл действительно скопирован и имеет размер
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    print(f"Файл успешно скопирован и имеет размер: {os.path.getsize(file_path)} байт")
                else:
                    print(f"Файл скопирован, но имеет нулевой размер или не существует")
                    return jsonify({"success": False, "error": "Файл скопирован, но имеет нулевой размер"}), 500
            else:
                print(f"Не удалось скопировать файл. Создаем заглушку.")
                
                # Создаем простой временный файл в формате .obj
                temp_obj_content = """
# Временный OBJ файл - заглушка
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
                
                # Сохраняем временный файл
                try:
                    with open(file_path, 'w') as f:
                        f.write(temp_obj_content)
                    print(f"Создана временная заглушка в {file_path}")
                except Exception as e:
                    print(f"Ошибка при создании заглушки: {str(e)}")
                    return jsonify({"success": False, "error": str(e)}), 500
        else:
            print(f"Файл существует, размер: {file_size} байт")
        
        # Отправляем файл клиенту
        print(f"Отправка файла модели: {file_path}")
        return send_file(file_path, mimetype='model/obj')
        
    except Exception as e:
        print(f"Ошибка при отправке модели: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download_model/<filename>')
def download_model(filename):
    """Скачивание модели в выбранном формате"""
    try:
        # Нормализуем имя файла, заменяя обратные слеши на прямые
        filename = normalize_path(filename)
        
        # Получаем запрошенный формат
        format = request.args.get('format', 'obj').lower()
        if format not in ['obj', 'stl', 'fbx', 'ply']:
            return jsonify({'success': False, 'message': 'Неподдерживаемый формат файла'}), 400

        # Проверяем наличие расширения в имени файла и удаляем его, если есть
        base_filename = filename
        if '.' in filename:
            base_filename = filename.split('.')[0]
            
        print(f"[DEBUG] Запрос на скачивание модели: {base_filename} в формате {format}")

        # Проверяем существование файла
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.obj")
        print(f"[DEBUG] Ищем файл по пути: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"[DEBUG] Файл не найден локально, пытаемся скачать с сервера генерации")
            # Если файла нет в локальном хранилище, пытаемся скачать его с сервера генерации
            response = download_model_file(f"{base_filename}.obj")
            if response.status_code != 200:
                print(f"[ERROR] Файл не найден на сервере генерации: {response.status_code}")
                return jsonify({'success': False, 'message': 'Файл не найден'}), 404
                
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}.obj")
            print(f"[DEBUG] Файл скачан с сервера генерации, новый путь: {file_path}")

        # Если пользователь авторизован, сохраняем информацию о скачивании
        if 'user_id' in session:
            user_id = session['user_id']
            prompt = request.args.get('prompt', '')
            
            # Проверяем, есть ли уже такая запись
            existing_download = DownloadedFile.query.filter_by(user_id=user_id, filename=f"{base_filename}.obj").first()
            if not existing_download:
                # Сохраняем новую запись о скачивании
                downloaded_file = DownloadedFile(
                    user_id=user_id,
                    filename=f"{base_filename}.obj",
                    prompt=prompt
                )
                db.session.add(downloaded_file)
                db.session.commit()
                print(f"[DEBUG] Сохранена информация о скачивании для пользователя {user_id}")

        # Загружаем модель с помощью trimesh
        print(f"[DEBUG] Загружаем модель с помощью trimesh: {file_path}")
        mesh = trimesh.load(file_path)
        
        # Конвертируем в запрошенный формат
        export_format = format.upper()
        export_data = BytesIO()
        
        print(f"[DEBUG] Конвертируем модель в формат {format}")
        if format == 'obj':
            export_data = mesh.export(file_type='obj')
        elif format == 'stl':
            export_data = mesh.export(file_type='stl')
        elif format == 'ply':
            export_data = mesh.export(file_type='ply')
        elif format == 'fbx':
            # Для FBX используем промежуточный формат
            temp_obj = BytesIO()
            mesh.export(file_obj=temp_obj, file_type='obj')
            temp_obj.seek(0)
            # Здесь можно добавить конвертацию в FBX, если есть библиотека
            # Пока что возвращаем OBJ с расширением FBX
            export_data = temp_obj.getvalue()

        # Создаем ответ с правильными заголовками
        response = make_response(export_data)
        response.headers['Content-Type'] = f'model/{format}'
        response.headers['Content-Disposition'] = f'attachment; filename={base_filename}.{format}'
        
        print(f"[DEBUG] Отправляем файл пользователю: {base_filename}.{format}")
        return response

    except Exception as e:
        print(f"[ERROR] Ошибка при скачивании модели: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/generation_progress', methods=['GET'])
def get_generation_progress():
    """
    Маршрут для получения текущего прогресса генерации модели.
    
    Returns:
        JSON с информацией о текущем прогрессе генерации
    """
    try:
        # Получаем информацию о прогрессе с локального ПК
        progress_data, status_code = check_generation_progress()
        
        # Если генерация запущена, добавляем информацию о тарифе пользователя
        if status_code == 200:
            # Добавляем описание тарифа
            subscription_level = progress_data.get('subscription_level', 1)
            
            subscription_names = {
                1: "Free", 
                2: "Pro", 
                3: "Business"
            }
            
            # Добавляем информацию о тарифе и его возможностях
            progress_data['subscription_name'] = subscription_names.get(subscription_level, "Free")
            
            # Добавляем описание качества модели в зависимости от тарифа
            if subscription_level == 1:
                progress_data['quality_description'] = "Базовое качество (тариф Free)"
            elif subscription_level == 2:
                progress_data['quality_description'] = "Улучшенное качество и детализация (тариф Pro)"
            elif subscription_level == 3:
                progress_data['quality_description'] = "Премиальное качество с максимальной детализацией (тариф Business)"
            else:
                progress_data['quality_description'] = "Базовое качество"
            
            # Если генерация завершена успешно, формируем URL для доступа к модели
            if progress_data.get('status') == 'completed':
                filename = progress_data.get('filename')
                if filename:
                    # Нормализуем имя файла, заменяя обратные слеши на прямые
                    filename = normalize_path(filename)
                    
                    # Формируем URL для модели
                    model_url = f"/models/{filename}"
                    progress_data['vds_model_url'] = model_url
                    
                    # URL для скачивания
                    download_url = f"/download_model/{filename}"
                    progress_data['vds_download_url'] = download_url
                    
                    # Также добавляем прямой URL для доступа к файлу на локальном сервере
                    direct_url = f"/proxy/direct/output/{filename}"
                    progress_data['direct_url'] = direct_url
                    
                    # Пытаемся скопировать файл с локального ПК
                    file_path = os.path.join(MODELS_FOLDER, secure_filename(filename))
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        # Файл уже скопирован на VDS
                        progress_data['file_ready'] = True
                        progress_data['file_path'] = f"/static/models/{filename}"
                    else:
                        # Файл еще не скопирован
                        progress_data['file_ready'] = False
                        
                        # Пробуем скопировать файл
                        if copy_model_from_local_server(filename):
                            progress_data['file_ready'] = True
                            progress_data['file_path'] = f"/static/models/{filename}"
        
        # Возвращаем информацию о прогрессе
        return jsonify(progress_data), status_code
            
    except Exception as e:
        error_message = f"Ошибка при получении прогресса генерации: {str(e)}"
        print(error_message)
        return jsonify({
            'status': 'error',
            'message': error_message,
            'progress': 0
        }), 500

@app.route('/check_confirmation', methods=['GET'])
def check_confirmation():
    """Проверка подтверждения аккаунта"""
    try:
        if 'user_id' not in session:
            return jsonify({'confirmed': False}), 200
            
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'confirmed': False}), 200
            
        return jsonify({
            'confirmed': bool(user.confirmed),
            'username': user.username
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке подтверждения: {str(e)}")
        return jsonify({
            'confirmed': False,
            'error': str(e)
        }), 500

@app.route('/verify_code', methods=['POST'])
def verify_code():
    """Проверка кода подтверждения, отправленного по email"""
    try:
        data = request.get_json()
        username = data.get('username')
        code = data.get('code')
        
        if not username or not code:
            return jsonify({
                'success': False,
                'message': 'Не указано имя пользователя или код подтверждения'
            }), 400
            
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({
                'success': False,
                'message': 'Пользователь не найден'
            }), 404
            
        # Проверяем код подтверждения
        if str(user.confirmation_code) == str(code):
            # Обновляем статус подтверждения
            user.confirmed = 1
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Регистрация успешно подтверждена'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Неверный код подтверждения'
            }), 400
            
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке кода подтверждения: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_random_password(length=12):
    """Генерация случайного пароля заданной длины"""
    import string
    import random
    
    # Символы для генерации пароля
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    
    # Генерируем пароль
    password = ''.join(random.choice(characters) for i in range(length))
    
    # Проверяем, что пароль содержит хотя бы одну цифру и один специальный символ
    while (not any(c.isdigit() for c in password) or 
           not any(c in "!@#$%^&*" for c in password)):
        password = ''.join(random.choice(characters) for i in range(length))
    
    return password

@app.route('/reset_password', methods=['POST'])
def reset_password():
    """Обработчик сброса пароля"""
    try:
        data = request.get_json()
        username = data.get('username')
        
        if not username:
            return jsonify({
                'success': False,
                'message': 'Не указано имя пользователя'
            }), 400
            
        # Находим пользователя
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({
                'success': False,
                'message': 'Пользователь не найден'
            }), 404
            
        # Генерируем новый пароль
        new_password = generate_random_password()
        
        # Хешируем новый пароль
        password_hash = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=16)
        
        # Обновляем пароль в базе данных
        user.password = password_hash
        db.session.commit()
        
        # Отправляем новый пароль на почту
        smtp_server = "smtp.yandex.ru"
        smtp_port = 465
        sender_email = "info.structo@yandex.ru"
        password = "quauawansqwxrrly"
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = user.email
            msg['Subject'] = "Восстановление пароля"
            
            body = f"""
            Здравствуйте, {username}!
            
            Вы запросили восстановление пароля на сайте Structo.
            Ваш новый пароль: {new_password}
            
            Рекомендуем сменить этот пароль после входа в систему.
            
            Если вы не запрашивали восстановление пароля, пожалуйста, немедленно свяжитесь с поддержкой.
            
            С уважением,
            Команда поддержки Pyan Inc.
            """
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender_email, password)
                server.send_message(msg)
                
            print(f"[DEBUG] Письмо с новым паролем успешно отправлено на {user.email}")
            
            return jsonify({
                'success': True,
                'message': 'Новый пароль отправлен на вашу почту'
            }), 200
            
        except Exception as e:
            print(f"[ERROR] Ошибка при отправке письма: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Ошибка при отправке письма с новым паролем'
            }), 500
            
    except Exception as e:
        print(f"[ERROR] Ошибка при сбросе пароля: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# Личный кабинет
# =============================================================================

@app.route('/account')
def account():
    """Страница личного кабинета пользователя"""
    if 'user_id' not in session:
        return redirect(url_for('main_page'))
    return render_template('account.html')

@app.route('/user_profile', methods=['GET'])
def user_profile():
    """API для получения данных профиля пользователя"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Не авторизован'}), 401
    
    try:
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'success': False, 'message': 'Пользователь не найден'}), 404
        
        return jsonify({
            'success': True,
            'username': user.username,
            'email': user.email,
            'display_name': user.display_name,
            'confirmed': user.confirmed == 1
        })
    except Exception as e:
        print(f"[ERROR] Ошибка при получении профиля: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/user_models', methods=['GET'])
def user_models():
    """API для получения списка моделей пользователя"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Не авторизован'}), 401
    
    try:
        user_id = session['user_id']
        models = DownloadedFile.query.filter_by(user_id=user_id).order_by(DownloadedFile.download_time.desc()).all()
        
        models_list = []
        for model in models:
            models_list.append({
                'id': model.id,
                'filename': model.filename,
                'prompt': model.prompt,
                'created_at': model.download_time.isoformat()
            })
        
        return jsonify({
            'success': True,
            'models': models_list
        })
    except Exception as e:
        print(f"[ERROR] Ошибка при получении моделей: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/change_password', methods=['POST'])
def change_password():
    """API для изменения пароля пользователя"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Не авторизован'}), 401
    
    try:
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'message': 'Необходимо указать текущий и новый пароль'}), 400
        
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'success': False, 'message': 'Пользователь не найден'}), 404
        
        # Проверка текущего пароля
        if not check_password_hash(user.password, current_password):
            return jsonify({'success': False, 'message': 'Неверный текущий пароль'}), 401
        
        # Хеширование и сохранение нового пароля
        password_hash = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=16)
        user.password = password_hash
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Пароль успешно изменен'})
    except Exception as e:
        print(f"[ERROR] Ошибка при изменении пароля: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/update_profile', methods=['POST'])
def update_profile():
    """API для обновления профиля пользователя"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Не авторизован'}), 401
    
    try:
        data = request.get_json()
        display_name = data.get('display_name')
        
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'success': False, 'message': 'Пользователь не найден'}), 404
        
        # Обновление отображаемого имени
        user.display_name = display_name
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Профиль успешно обновлен',
            'username': user.username,
            'email': user.email,
            'display_name': user.display_name
        })
    except Exception as e:
        print(f"[ERROR] Ошибка при обновлении профиля: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/subscriptions')
def subscriptions():
    if 'user_id' not in session:
        return redirect(url_for('main_page'))
    user = User.query.get(session['user_id'])
    subscription = user.subscription if user else 1
    return render_template('subscriptions.html', subscription=subscription)

@app.route('/pay')
def pay():
    plan = request.args.get('plan', '')
    return render_template('pay.html', plan=plan)

@app.route('/api/subscription')
def api_subscription():
    if 'user_id' not in session:
        return jsonify({'subscription': 1})
    user = User.query.get(session['user_id'])
    return jsonify({'subscription': user.subscription if user else 1})

@app.route('/proxy/download/<filename>')
def proxy_download(filename):
    """
    Маршрут для скачивания модели с локального сервера и предоставления URL для доступа к ней.
    
    Args:
        filename (str): Имя файла модели
        
    Returns:
        JSON с URL для доступа к модели
    """
    try:
        # Нормализуем имя файла, заменяя обратные слеши на прямые
        filename = normalize_path(filename)
        
        print(f"Запрос на копирование файла с локального сервера: {filename}")
        
        # Копируем файл с локального сервера
        if copy_model_from_local_server(filename):
            # Проверяем, что файл действительно скопирован
            file_path = os.path.join(MODELS_FOLDER, secure_filename(filename))
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"Файл успешно скопирован и доступен: {file_path}, размер: {os.path.getsize(file_path)} байт")
                
                # Формируем URL для доступа к модели из браузера
                model_url = f"/models/{filename}"
                download_url = f"/download_model/{filename}"
                
                return jsonify({
                    'status': 'success',
                    'message': 'Файл успешно скопирован',
                    'vds_model_url': model_url,
                    'vds_download_url': download_url
                })
            else:
                print(f"Файл не был успешно скопирован или имеет нулевой размер")
                return jsonify({
                    'status': 'error',
                    'message': 'Файл не удалось скопировать или он имеет нулевой размер'
                }), 500
        else:
            print(f"Не удалось скопировать файл с локального сервера")
            return jsonify({
                'status': 'error',
                'message': 'Не удалось скопировать файл с локального сервера'
            }), 500
            
    except Exception as e:
        print(f"Ошибка при копировании файла: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/static/models/<filename>')
def serve_static_model(filename):
    """
    Маршрут для отдачи файлов моделей из папки static/models.
    Дополнительный маршрут для доступа к моделям по прямому пути.
    """
    try:
        print(f"Запрос на получение статической модели: {filename}")
        file_path = os.path.join(MODELS_FOLDER, secure_filename(filename))
        
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            return jsonify({'error': 'Файл не найден'}), 404
            
        print(f"Отправка файла: {file_path}, размер: {os.path.getsize(file_path)} байт")
        return send_file(file_path, mimetype='model/obj')
    except Exception as e:
        print(f"Ошибка при отправке статической модели: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/proxy/direct/<path:file_path>')
def proxy_direct_to_local_server(file_path):
    """
    Прямое проксирование запроса к локальному серверу.
    Особенно полезно для доступа к файлам в директории output/.
    
    Args:
        file_path (str): Путь к файлу на локальном сервере
        
    Returns:
        Содержимое файла или сообщение об ошибке
    """
    try:
        # Нормализуем путь к файлу, заменяя обратные слеши на прямые
        file_path = normalize_path(file_path)
        
        from config import LOCAL_PC_IP, LOCAL_PC_PORT
        
        # Формируем URL для локального сервера
        url = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/{file_path}"
        print(f"[PROXY] Перенаправление запроса к локальному серверу: {url}")
        
        # Отправляем GET запрос
        response = requests.get(url, timeout=30, stream=True)
        print(f"[PROXY] Ответ получен. Статус: {response.status_code}")
        
        if response.status_code == 200:
            # Создаем ответ Flask на основе полученного ответа
            headers = [(name, value) for name, value in response.headers.items() 
                      if name.lower() not in ('content-encoding', 'content-length', 'transfer-encoding', 'connection')]
            
            # Определяем правильный тип содержимого
            content_type = response.headers.get('content-type', 'application/octet-stream')
            
            # Возвращаем содержимое файла клиенту
            return Response(response.iter_content(chunk_size=10*1024),
                           status=response.status_code,
                           headers=headers,
                           content_type=content_type)
        else:
            # Возвращаем ошибку, если файл не найден или другая проблема
            error_message = f"Ошибка при запросе к локальному серверу: HTTP {response.status_code}"
            print(f"[PROXY] {error_message}")
            return jsonify({"status": "error", "message": error_message}), response.status_code
            
    except requests.exceptions.ConnectionError as e:
        error_message = f"Ошибка соединения с локальным сервером: {str(e)}"
        print(f"[PROXY] {error_message}")
        return jsonify({"status": "error", "message": error_message}), 502
        
    except requests.exceptions.Timeout as e:
        error_message = f"Таймаут при соединении с локальным сервером: {str(e)}"
        print(f"[PROXY] {error_message}")
        return jsonify({"status": "error", "message": error_message}), 504
        
    except Exception as e:
        error_message = f"Непредвиденная ошибка: {str(e)}"
        print(f"[PROXY] {error_message}")
        return jsonify({"status": "error", "message": error_message}), 500

# =============================================================================
# Главный блок запуска приложения
# =============================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Обеспечиваем доступ извне при запуске на VDS
    app.run(host='0.0.0.0', port=5001)  # Отключаем SSL для Flask
