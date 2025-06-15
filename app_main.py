#!/usr/bin/env python3
# Основной запускающий скрипт
import os
import sys

# Добавляем текущую директорию в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем app из основного файла
from app import app

if __name__ == '__main__':
    # Запускаем основное приложение на порту 5000
    app.run(host='0.0.0.0', port=5678)
