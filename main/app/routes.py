from flask import Blueprint, request, render_template, send_from_directory, url_for, jsonify, session, current_app
import os
import argparse
from datetime import datetime
from utils.conditional_generation import generate as conditional_generate

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

# Страница генерации
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

    parser = argparse.ArgumentParser(description="VAE-GAN for text-to-3D Model Generation on custom dataset")
    parser.add_argument("--dataset", type=str, default="testdataset", help="Path to dataset directory containing OFF files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="vae_gan_test.pth", help="Checkpoint file path")
    parser.add_argument("--voxel_size", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--output", type=str, default="static/models/model.ply", help="Output PLY file for generation")
    parser.add_argument("--lambda_adv", type=float, default=0.001, help="Weight factor for adversarial loss")
    parser.add_argument("--prompt", type=str, default=text, help="Text prompt for conditional 3D generation")
    parser.add_argument("--cond_dim", type=int, default=512, help="Dimension of the condition vector")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of latent vector")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--mode", type=str, default='generate', help="Mode of operation: 'generate'")
    
    # Парсим пустой список, т.к. не берём аргументы из CLI
    args = parser.parse_args([])
    # Указываем путь к чекпоинту (ваш .pth файл)
    args.checkpoint = "models/checkpoints/vae_gan_test.pth"
    
    # Указываем абсолютный путь, где будет сохранён результат (model.ply)
    args.output = os.path.join(current_app.root_path, 'static', 'models', 'model.ply')

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Генерация
    scene_filename = conditional_generate(args=args, device=device, need_visualisation=False)
    
    # Возвращаем JSON: 'model_url' с маршрутом скачивания 
    filename = os.path.basename(scene_filename)  # model.ply
    # Ссылка на скачивание через наше download_file
    model_url = url_for('main.download_file', filename=filename)
    print(model_url)
    return jsonify({'model_url': model_url})

# Маршрут для скачивания сгенерированного .ply
@main_bp.route('/static/models/<filename>')
def download_file(filename):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Вы должны быть авторизованы для скачивания файлов.'}), 403

    # Путь в app.config['UPLOAD_FOLDER']? Или проще path = os.path.join(current_app.root_path, 'static', 'models')
    folder_path = os.path.join(current_app.root_path, 'static', 'models')
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Файл не найден.'}), 404

    from app.db_models import DownloadedFile
    from app import db
    downloaded_file = DownloadedFile(user_id=user_id, filename=filename)
    db.session.add(downloaded_file)
    db.session.commit()

    # Отдаём как "attachment" — если хотите просто открывать, можно убрать as_attachment
    return send_from_directory(folder_path, filename, as_attachment=False, mimetype='application/octet-stream')

@main_bp.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    lang = data.get('lang', 'en')
    session['lang'] = lang
    return jsonify({'success': True})
