from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import os
import torch
from embedding_generator import EmbeddingGenerator
from model_generator import generate_3d_scene_from_embedding
from gan_model import Generator, Discriminator
import numpy as np

from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
UPLOAD_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Инициализация устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство для веб-приложения: {device}")

# Инициализация генератора эмбеддингов
embedding_generator = EmbeddingGenerator(device, reduced_dim=512)

# Инициализация GAN
input_dim = 512
output_dim = 512
generator = Generator(input_dim=input_dim, output_dim=output_dim).to(device)
discriminator = Discriminator(input_dim=output_dim).to(device)

generator_path = 'generator.pth'
discriminator_path = 'discriminator.pth'

if os.path.exists(generator_path) and os.path.exists(discriminator_path):
    generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device, weights_only=True))
    generator.eval()
    discriminator.eval()
    print("Модели GAN успешно загружены.")
else:
    print("Модели GAN не найдены. Убедитесь, что 'generator.pth' и 'discriminator.pth' существуют.")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Текстовое поле пустое. Пожалуйста, введите текст.'}), 400
    print(f"[APP] Получен запрос на генерацию модели для текста: {text}")
    
    embedding = embedding_generator.generate_embedding(text).squeeze()
    
    with torch.no_grad():
        noise = torch.randn(1, input_dim).to(device)
        generated_embedding = generator(noise).cpu().numpy().squeeze()
    
    print("[APP] Генерируем 3D модель на основе эмбеддинга...")
    scene_filename = generate_3d_scene_from_embedding(generated_embedding, text)
    
    filename = os.path.basename(scene_filename)
    model_url = url_for('download_file', filename=filename)
    return jsonify({'model_url': model_url})

@app.route('/downloads/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Файл не найден.'}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True, mimetype='application/octet-stream')

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/prompt_guide')
def prompt_guide():
    return render_template('prompt_guide.html')

@app.route('/about')
def about():
    return render_template('about.html')



# Настройка базы данных SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key_here'
db = SQLAlchemy(app)

# Модель пользователя
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)

@app.route('/try-title')
def main_page():
    return render_template('main_page.html')


@app.route('/auth_status', methods=['GET'])
def auth_status():
    user_id = session.get('user_id')
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

        # Исправленный запрос для поиска пользователя по имени
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Неверный логин или пароль'}), 401


    elif action == 'register':
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        # Проверка на существование пользователя
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Пользователь с таким логином уже существует'}), 400

        # Хеширование пароля
        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

        new_user = User(username=username, password=password_hash, email=email)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'success': True})



@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)  # Удаляем данные пользователя из сессии
    return jsonify({'success': True})


@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get('lang', 'en')  # По умолчанию английский
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
    lang = session.get('lang', 'ru')  # По умолчанию русский
    return translations[lang].get(key, key)





if __name__ == '__main__':
    # Создание базы данных (если ее еще нет)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
