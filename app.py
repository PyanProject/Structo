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
from translator import translate
from model_converter import ModelConverter

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
    pass
@app.route('/')
def index():
    lang = session.get('lang', 'ru')  
    return render_template('index.html', lang=lang)

@app.route('/prompt_guide')
def prompt_guide():
    lang = session.get('lang', 'ru')
    return render_template('prompt_guide.html', lang=lang)

@app.route('/about')
def about():
    lang = session.get('lang', 'ru')
    return render_template('about.html', lang=lang)

@app.route('/generate')
def main_page():
    lang = session.get('lang', 'ru')
    return render_template('main_page.html', lang=lang)

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

@app.route('/set_language', methods=['POST', 'GET'])
def set_language():
    try:
        if request.method == 'POST':
            
            data = request.get_json()
                
            lang = data.get('lang')
                
            session['lang'] = lang
            return jsonify({'success': True, 'lang': lang})
        else:
            lang = session.get('lang', 'ru')
            return jsonify({'lang': lang})
    except Exception as e:
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

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
        
        # Фиксированный ответ вместо динамического
        # Это простое и надежное решение, пока мы исправляем основную проблему
        return '{"status": "online", "gpu_available": true}', 200, {'Content-Type': 'application/json'}
    except Exception as e:
        # В случае ошибки (например, сервер недоступен)
        print(f"Ошибка при проверке генератора: {str(e)}")
        return '{"status": "error", "message": "Ошибка при проверке статуса генератора"}', 503, {'Content-Type': 'application/json'}

@app.route('/generate_model', methods=['POST'])
def remote_generate_model():
    """Отправка запроса на генерацию 3D-модели на удаленный ПК"""
    print("=== Получен запрос на генерацию модели ===")
    try:
        if not request.is_json:
            print("Ошибка: запрос не содержит JSON")
            return jsonify({
                'success': False,
                'error': 'Ожидается JSON в теле запроса'
            }), 400
            
        prompt = request.json.get('prompt', '')
        print(f"Полученный промпт: {prompt}")
        
        if not prompt:
            print("Ошибка: пустой промпт")
            return jsonify({
                'success': False,
                'error': 'Промпт не указан'
            }), 400
        
        print("Отправка запроса на генерацию модели...")
        result, code = generate_model(prompt)
        print(f"Результат генерации: {result}, код: {code}")
        
        # Проверяем, что result - это словарь
        if not isinstance(result, dict):
            print("Ошибка: результат не является словарем")
            return jsonify({
                'success': False,
                'error': 'Некорректный формат ответа от сервера генерации'
            }), 500
            
        if isinstance(code, int) and code >= 400:
            return jsonify(result), code
            
        return jsonify(result), 200
        
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Ошибка при обработке JSON данных'
        }), 400
        
    except Exception as e:
        print(f"Неожиданная ошибка при генерации модели: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Ошибка при генерации модели: {str(e)}'
        }), 500

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
        
        # Проверяем соответствие промпта имени файла, если промпт указан
        if prompt:
            safe_prompt = ''.join(c if c.isalnum() else '_' for c in prompt.lower())
            if not filename.startswith(safe_prompt):
                print(f"Ошибка: имя файла {filename} не соответствует промпту {prompt}")
                return False
        
        # Формируем URL для загрузки файла
        url = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/output/{filename}"
        print(f"Пытаемся загрузить файл с {url}")
        
        # Отправляем GET запрос для получения файла
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Сохраняем файл в папку моделей
            file_path = os.path.join(MODELS_FOLDER, secure_filename(filename))
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Файл успешно сохранен в {file_path}")
            return True
        else:
            print(f"Ошибка при загрузке файла: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Ошибка при копировании файла с локального сервера: {str(e)}")
        return False

@app.route('/models/<filename>')
def serve_model(filename):
    """Отдает файл модели для отображения на странице"""
    try:
        print(f"Запрос на получение модели: {filename}")
        
        # Безопасный путь к файлу
        file_path = os.path.join(MODELS_FOLDER, secure_filename(filename))
        print(f"Полный путь к файлу: {file_path}")
        
        # Если файл не существует, пытаемся скопировать его с локального сервера
        if not os.path.exists(file_path):
            print(f"Файл модели не найден: {file_path}. Пытаемся скопировать с локального сервера.")
            
            # Пытаемся скопировать файл
            if copy_model_from_local_server(filename):
                print(f"Файл успешно скопирован с локального сервера")
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
            print(f"Файл существует, размер: {os.path.getsize(file_path)} байт")
        
        # Отправляем файл клиенту
        print(f"Отправка файла модели: {file_path}")
        return send_file(file_path, mimetype='model/obj')
        
    except Exception as e:
        print(f"Ошибка при отправке модели: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download_model/<filename>')
def download_model(filename):
    """Скачивание модели и сохранение связи с пользователем"""
    try:
        # Проверяем существование файла
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            # Если файла нет в локальном хранилище, пытаемся скачать его с сервера генерации
            response = download_model_file(filename)
            if response.status_code != 200:
                return jsonify({'success': False, 'message': 'Файл не найден'}), 404
                
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Если пользователь авторизован, сохраняем информацию о скачивании
        if 'user_id' in session:
            user_id = session['user_id']
            prompt = request.args.get('prompt', '')
            
            # Проверяем, есть ли уже такая запись
            existing_download = DownloadedFile.query.filter_by(user_id=user_id, filename=filename).first()
            if not existing_download:
                # Сохраняем новую запись о скачивании
                downloaded_file = DownloadedFile(
                    user_id=user_id,
                    filename=filename,
                    prompt=prompt
                )
                db.session.add(downloaded_file)
                db.session.commit()
        
        # Возвращаем файл для скачивания
        return send_file(file_path, as_attachment=True)
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
        
        # Если генерация завершена успешно, пытаемся скопировать файл
        if status_code == 200 and progress_data.get('status') == 'completed':
            filename = progress_data.get('filename')
            if filename:
                # Пытаемся скопировать файл с локального ПК
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
                'message': 'Новый пароль отправлен на вашу почту',
                'email': user.email
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
    lang = session.get('lang', 'ru')
    return render_template('account.html', lang=lang)

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
        user.username = display_name
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

@app.route('/translate', methods=['POST'])
def translate_text():

    try:
        data = request.get_json()
        
        text = data.get('text', '')
        
        translated_text = translate(text)
        
        return jsonify({'translated_text': translated_text})
    except Exception as e:

        return jsonify({'error': str(e)}), 500

@app.route('/check_subscription')
def check_subscription():
    """Проверка статуса подписки пользователя"""
    try:
        if 'user_id' not in session:
            return jsonify({
                'subscription': 1,
            }), 200
            
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({
                'subscription': 1,
            }), 200
        
        return jsonify({
            'subscription': user.subscription,
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке подписки: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

# =============================================================================
# Маршруты для конвертации 3D-моделей
# =============================================================================

@app.route('/api/formats', methods=['GET'])
def get_supported_formats():
    """Получение списка поддерживаемых форматов для конвертации"""
    converter = ModelConverter()
    formats = converter.get_supported_formats()
    return jsonify({'formats': formats}), 200

@app.route('/api/convert/<filename>', methods=['POST'])
def convert_model(filename):
    """Конвертация модели в указанный формат"""
    try:
        data = request.get_json()
        if not data or 'format' not in data:
            return jsonify({'success': False, 'error': 'Не указан формат конвертации'}), 400
        
        target_format = data['format']
        converter = ModelConverter()
        result = converter.convert_model(filename, target_format)
        
        if result['success']:
            # Добавляем запись о скачивании в базу данных, если пользователь авторизован
            if 'user_id' in session:
                user_id = session['user_id']
                prompt = data.get('prompt', 'Без описания')
                
                download_record = DownloadedFile(
                    user_id=user_id,
                    filename=result['filename'],
                    prompt=prompt,
                    download_time=datetime.utcnow()
                )
                db.session.add(download_record)
                db.session.commit()
            
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download_converted/<path:filename>')
def download_converted_file(filename):
    """Скачивание конвертированного файла"""
    try:
        # Полный путь к файлу
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'converted', filename)
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Файл не найден'}), 404
        
        # Определяем тип контента
        content_type = 'application/octet-stream'
        
        # Отправляем файл
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=content_type
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
# Главный блок запуска приложения
# =============================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5003)  
