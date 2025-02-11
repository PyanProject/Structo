from flask import Blueprint, request, render_template, send_from_directory, url_for, jsonify, session, current_app
import os
import torch
from utils.mesh_generator import generate_3d_scene_from_embedding

main_bp = Blueprint('main', __name__)

# Главная страница
@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/prompt_guide')
def prompt_guide():
    return render_template('prompt_guide.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

# Страница с формой генерации
@main_bp.route('/generate', methods=['GET'])
def main_page():
    return render_template('main_page.html')

# API для генерации 3D-модели (POST)
@main_bp.route('/api/generate', methods=['POST'])
def generate_model():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Вы должны быть авторизованы для генерации моделей.'}), 403

    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Текстовое поле пустое. Пожалуйста, введите текст.'}), 400
    print(f"[APP] Получен запрос на генерацию модели для текста: {text}")

    # Получаем из контекста приложения необходимые объекты
    device = current_app.config.get('DEVICE')
    embedding_generator = current_app.embedding_generator
    generator = current_app.generator

    # Генерируем эмбеддинг на основе текста
    # Генерируем эмбеддинг на основе текста
    embedding = embedding_generator.generate_embedding(text).squeeze()
    if embedding.dim() == 1:  # если эмбеддинг имеет размерность (512,), делаем батч из одного элемента
        embedding = embedding.unsqueeze(0)

    with torch.no_grad():
        noise = torch.randn(1, generator.noise_dim).to(device)
        generated_embedding = generator(noise, embedding).cpu().numpy().squeeze()

    print("[APP] Генерируем 3D модель на основе эмбеддинга...")
    # Генерация 3D-сцены из сгенерированного эмбеддинга и текста
    scene_filename = generate_3d_scene_from_embedding(generated_embedding, text)

    # Формируем URL для скачивания (файл сохраняется в static/models)
    filename = os.path.basename(scene_filename)  # Например, model.obj
    model_url = url_for('main.download_file', filename=filename)
    return jsonify({'model_url': model_url})

# Маршрут для скачивания сгенерированного .obj файла
@main_bp.route('/static/models/<filename>')
def download_file(filename):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Вы должны быть авторизованы для скачивания файлов.'}), 403

    folder_path = os.path.join(current_app.root_path, 'static', 'models')
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Файл не найден.'}), 404

    # Регистрируем скачивание в базе данных
    from app.db_models import DownloadedFile
    from app import db
    downloaded_file = DownloadedFile(user_id=user_id, filename=filename)
    db.session.add(downloaded_file)
    db.session.commit()

    # Отдаём файл как attachment (если нужно скачать)
    return send_from_directory(folder_path, filename, as_attachment=True, mimetype='application/octet-stream')

@main_bp.route('/set_language', methods=['POST'])
def set_language_route():
    data = request.get_json()
    lang = data.get('lang', 'en')
    session['lang'] = lang
    return jsonify({'success': True})
