import torch
import os
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_config

print("Проверка доступности обучения модели Shap-E")

# Проверяем GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Загружаем конфигурацию диффузии
print("Загрузка конфигурации диффузии...")
try:
    diffusion_config = load_config('diffusion')
    print("Конфигурация диффузии успешно загружена")
    print(f"Тип диффузии: {diffusion_config.get('model_mean_type', 'не указан')}")
    print(f"Расписание бета: {diffusion_config.get('beta_schedule', 'не указано')}")
    
    # Создаем объект диффузии
    print("Создание модели диффузии...")
    diffusion = diffusion_from_config(diffusion_config)
    print("Модель диффузии успешно создана!")
    
    # Проверка путей и параметров для обучения
    print("\nНеобходимые шаги для обучения:")
    print("1. Подготовить датасет в правильном формате (изображения, облака точек, метаданные)")
    print("2. Создать конфигурационный файл обучения")
    print("3. Указать пути к данным в конфигурационном файле")
    print("4. Запустить обучение с помощью соответствующего скрипта")
    
    print("\nПример датасета:")
    print("- Изображения с разных ракурсов (.png)")
    print("- Облака точек для каждого объекта")
    print("- Текстовые описания (для текстово-условной модели)")
    
except Exception as e:
    print(f"Ошибка при инициализации: {str(e)}") 