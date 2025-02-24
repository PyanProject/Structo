from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, session
import os
import argparse
import sqlite3
from datetime import datetime, timedelta

import torch
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Импортируем внешние модули
from embedding_generator import EmbeddingGenerator
from gan_model import Generator, Discriminator
from main import generate as newgen

# =============================================================================
# Инициализация приложения и конфигурация
# =============================================================================

# Создаем приложение с относительной конфигурацией

app = Flask(__name__, instance_relative_config=True)

# Задаем путь для загрузки сгенерированных моделей
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'models')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Создаем папку instance для базы данных, если её ещё нет
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.path.join(app.instance_path, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DB_PATH

# Другие настройки приложения
app.config['SECRET_KEY'] = 'your_secret_key_here'
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

class DownloadedFile(db.Model):
    """Модель загруженного файла."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(150), nullable=False)
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
# Инициализация внешних моделей и устройств
# =============================================================================

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Используемое устройство для веб-приложения: {device}")

# Инициализация генератора эмбеддингов с использованием CLIP
embedding_generator = EmbeddingGenerator(device, reduced_dim=512)

# Инициализация GAN-моделей
generator = Generator(noise_dim=100, embedding_dim=3072).to(device)
discriminator = Discriminator(data_dim=3072).to(device)

# Пути к сохраненным весам
generator_path = 'generator.pth'
discriminator_path = 'discriminator.pth'

if os.path.exists(generator_path) and os.path.exists(discriminator_path):
    generator.load_state_dict(torch.load(generator_path, map_location=device), strict=False)
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    generator.eval()
    discriminator.eval()
    print("Модели GAN успешно загружены.")
else:
    print("Модели GAN не найдены. Убедитесь, что 'generator.pth' и 'discriminator.pth' существуют.")

# =============================================================================
# Маршруты для генерации и скачивания моделей
# =============================================================================

@app.route('/generate', methods=['POST'])
def generate_model():
    """Генерирует 3D модель по текстовому описанию."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Вы должны быть авторизованы для генерации моделей.'}), 403

    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Текстовое поле пустое. Пожалуйста, введите текст.'}), 400

    print(f"[APP] Получен запрос на генерацию модели для текста: {text}")

    # Парсер аргументов (используется для интеграции с main.py)
    parser = argparse.ArgumentParser(
        description="VAE-GAN for text-to-3D Model Generation on custom dataset"
    )
    parser.add_argument("--dataset", type=str, default="testdataset",
                        help="Path to dataset directory containing OFF files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="vae_gan_test.pth",
                        help="Checkpoint file path")
    parser.add_argument("--voxel_size", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--output", type=str, default="static/models/model.obj",
                        help="Output OBJ file for generation")
    parser.add_argument("--lambda_adv", type=float, default=0.001,
                        help="Weight factor for adversarial loss")
    parser.add_argument("--prompt", type=str, default=text,
                        help="Text prompt for conditional 3D generation")
    parser.add_argument("--cond_dim", type=int, default=512,
                        help="Dimension of the condition vector")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Dimension of latent vector")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--mode", type=str, default='generate',
                        help="Mode of operation: 'generate'")
    
    args = parser.parse_args()
    args.checkpoint = "vae_gan_test.pth"  # Переопределяем путь к контрольной точке

    scene_filename = newgen(args=args, device=device, need_visualisation=False)
    filename = os.path.basename(scene_filename)
    model_url = url_for('download_file', filename=filename)
    print(model_url)
    return jsonify({'model_url': model_url})

@app.route('/static/models/<filename>')
def download_file(filename):
    """Отдает файл модели для скачивания, сохраняя информацию в БД."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Вы должны быть авторизованы для скачивания файлов.'}), 403

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Файл не найден.'}), 404

    downloaded_file = DownloadedFile(user_id=user_id, filename=filename)
    db.session.add(downloaded_file)
    db.session.commit()

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True,
                               mimetype='application/octet-stream')

# =============================================================================
# Основные страницы
# =============================================================================

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

# =============================================================================
# Маршруты для аутентификации
# =============================================================================

@app.route('/auth_status', methods=['GET'])
def auth_status():
    user_id = session.get('user_id')
    print(f"[DEBUG] Проверка сессии: user_id={user_id}, session.permanent={session.permanent}")
    if user_id:
        user = User.query.get(user_id)
        if user:
            return jsonify({'authenticated': True, 'username': user.username})
    return jsonify({'authenticated': False})

@app.route('/auth', methods=['POST'])
def auth():
    data = request.get_json()
    action = data.get('action')

    if action == 'login':
        username = data.get('username')
        password = data.get('password')
        remember_me = data.get('remember_me', False)
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session.permanent = remember_me
            print(f"[DEBUG] Успешный вход: user_id={user.id}, remember_me={remember_me}, "
                  f"session.permanent={session.permanent}")
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Неверный логин или пароль'}), 401

    elif action == 'register':
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Пользователь с таким логином уже существует'}), 400

        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        new_user = User(username=username, password=password_hash, email=email)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True})

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
# Главный блок запуска приложения
# =============================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
