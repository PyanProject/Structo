import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config.config import Config
import torch
from utils.clip_embedding import EmbeddingGenerator
from utils.gan_models import Generator, Discriminator

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Создаём необходимые папки (UPLOAD_FOLDER, статические ассеты, модели)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'static', 'assets'), exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'static', 'models'), exist_ok=True)
    
    # Инициализируем базу данных
    db.init_app(app)
    with app.app_context():
        db.create_all()  # Создадим таблицы, если их ещё нет

    # Инициализация вычислительного устройства (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.config['DEVICE'] = device
    print(f"Using device: {device}")

    # Инициализация генератора эмбеддингов
    embedding_generator = EmbeddingGenerator(device, reduced_dim=512)

    # Инициализация GAN-моделей
    noise_dim = 100        # корректное имя для первого параметра генератора
    embedding_dim = 512
    generator = Generator(noise_dim=noise_dim, embedding_dim=embedding_dim).to(device)
    discriminator = Discriminator(data_dim=12288, embedding_dim=embedding_dim).to(device)
        
    # Пути к файлам весов (предполагается, что они лежат в папке models относительно app.root_path)
    generator_path = os.path.join(app.root_path, 'models', 'generator.pth')
    discriminator_path = os.path.join(app.root_path, 'models', 'discriminator.pth')
    
    if os.path.exists(generator_path) and os.path.exists(discriminator_path):
        try:
            # Загружаем state_dict для генератора
            gen_state_dict = torch.load(generator_path, map_location=device, weights_only=True)
            # Если ключи имеют префикс "model.", удаляем его:
            new_gen_state_dict = {}
            for key, value in gen_state_dict.items():
                if key.startswith("model."):
                    new_key = key[len("model."):]
                else:
                    new_key = key
                new_gen_state_dict[new_key] = value
            generator.load_state_dict(new_gen_state_dict, strict=False)

            # Аналогично для дискриминатора
            disc_state_dict = torch.load(discriminator_path, map_location=device, weights_only=True)
            new_disc_state_dict = {}
            for key, value in disc_state_dict.items():
                if key.startswith("model."):
                    new_key = key[len("model."):]
                else:
                    new_key = key
                new_disc_state_dict[new_key] = value
            discriminator.load_state_dict(new_disc_state_dict, strict=False)

            generator.eval()
            discriminator.eval()
            print("GAN models loaded successfully.")
        except Exception as e:
            print(f"Error loading GAN models: {e}")
    else:
        print("GAN models not found. Ensure 'generator.pth' and 'discriminator.pth' exist.")
    
    # Сохраним объекты в контексте приложения для доступа в маршрутах
    app.embedding_generator = embedding_generator
    app.generator = generator
    app.discriminator = discriminator

    # Регистрируем blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    from app.auth import auth_bp
    app.register_blueprint(auth_bp)
    
    return app
