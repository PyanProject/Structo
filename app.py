# app.py

from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import os
import torch
from embedding_generator import EmbeddingGenerator
from model_generator import generate_3d_scene_from_embedding
from gan_model import Generator, Discriminator
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Инициализация устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство для веб-приложения: {device}")

# Инициализация генератора эмбеддингов
embedding_generator = EmbeddingGenerator(device, reduced_dim=512)

# Инициализация GAN
input_dim = 512  # Размерность эмбеддинга CLIP
output_dim = 512
generator = Generator(input_dim=input_dim, output_dim=output_dim).to(device)
discriminator = Discriminator(input_dim=output_dim).to(device)

# Загрузка обученных моделей
generator_path = 'generator.pth'
discriminator_path = 'discriminator.pth'

# app.py (обновлённые строки загрузки моделей)

if os.path.exists(generator_path) and os.path.exists(discriminator_path):
    generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device, weights_only=True))
    generator.eval()
    discriminator.eval()
    print("Модели GAN успешно загружены.")
else:
    print("Модели GAN не найдены. Убедитесь, что 'generator.pth' и 'discriminator.pth' существуют.")
    # Можно добавить логику для обучения GAN здесь или сообщить пользователю


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Текстовое поле пустое. Пожалуйста, введите текст.'}), 400
    print(f"Получен запрос на генерацию модели для текста: {text}")
    
    # Генерация эмбеддинга
    embedding = embedding_generator.generate_embedding(text).squeeze()
    
    # Генерация 3D эмбеддинга через GAN
    with torch.no_grad():
        noise = torch.randn(1, input_dim).to(device)
        generated_embedding = generator(noise).cpu().numpy().squeeze()
    
    # Генерация 3D модели
    scene_filename = generate_3d_scene_from_embedding(generated_embedding, text)
    
    # Получение имени файла
    filename = os.path.basename(scene_filename)
    
    # Возвращаем URL для загрузки и визуализации модели
    model_url = url_for('download_file', filename=filename)
    return jsonify({'model_url': model_url})

# app.py (изменённая часть)

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True, mimetype='application/octet-stream')

if __name__ == '__main__':
    app.run(debug=True)
