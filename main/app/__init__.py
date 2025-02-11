#__init__.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config.config import Config

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Создаём папки для статических файлов (если их ещё нет)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'static', 'assets'), exist_ok=True)
    
    # Инициализируем базу данных
    db.init_app(app)
    
    with app.app_context():
        db.create_all()  # Создадим таблицы (User, DownloadedFile, и т.д.) если их ещё нет
    
    
    # Регистрируем blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    from app.auth import auth_bp
    app.register_blueprint(auth_bp)
    
    return app
