#!/usr/bin/env python3
from app import app, db, User, DownloadedFile
import datetime

# Создаем контекст приложения
app.app_context().push()

# Ищем первого пользователя
user = User.query.first()

if user:
    print(f'Найден пользователь: {user.username}')
    
    # Добавляем тестовую модель для этого пользователя
    model = DownloadedFile(
        user_id=user.id, 
        filename='test_model.obj', 
        prompt='Тестовый красный куб', 
        download_time=datetime.datetime.now()
    )
    
    db.session.add(model)
    db.session.commit()
    print('Тестовая модель добавлена')
    
    # Выводим все модели пользователя
    models = DownloadedFile.query.filter_by(user_id=user.id).all()
    print(f'Всего моделей у пользователя: {len(models)}')
    for model in models:
        print(f' - {model.filename}: {model.prompt} ({model.download_time})')
else:
    print('Пользователей не найдено. Создайте пользователя через веб-интерфейс.') 