import os
from datetime import timedelta

# Определяем корневую директорию проекта динамически
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'users.db')
    SECRET_KEY = 'your_secret_key_here'
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)
    SESSION_COOKIE_DURATION = timedelta(days=30)
    # Папка для сохранения моделей – относительно корневой директории
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'models')
