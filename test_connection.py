#!/usr/bin/env python3
"""
Скрипт для тестирования соединения с генератором 3D-моделей на локальном ПК.
Запустите этот скрипт для проверки доступности API.
"""

import requests
import json
import time
import argparse
from config import HEALTH_CHECK_URL, GENERATION_API_URL, REQUEST_TIMEOUT

def check_generator_status():
    """Проверка статуса генератора"""
    try:
        print(f"Проверка соединения с {HEALTH_CHECK_URL}...")
        start_time = time.time()
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        elapsed = time.time() - start_time
        
        print(f"Статус ответа: {response.status_code}")
        print(f"Время ответа: {elapsed:.2f} секунд")
        
        if response.status_code == 200:
            data = response.json()
            print("Генератор моделей онлайн.")
            print(f"GPU доступен: {data.get('gpu_available', False)}")
            return True
        else:
            print(f"Ошибка: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения: {e}")
        return False

def test_model_generation(prompt):
    """Тестирование генерации модели"""
    if not check_generator_status():
        print("Генератор недоступен. Невозможно выполнить тест генерации.")
        return False
    
    try:
        print(f"\nОтправка запроса на генерацию модели с промптом: '{prompt}'")
        print(f"URL: {GENERATION_API_URL}")
        
        start_time = time.time()
        response = requests.post(
            GENERATION_API_URL,
            json={'prompt': prompt},
            timeout=REQUEST_TIMEOUT
        )
        elapsed = time.time() - start_time
        
        print(f"Статус ответа: {response.status_code}")
        print(f"Время генерации: {elapsed:.2f} секунд")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                print("Генерация успешна!")
                print(f"Имя файла: {data.get('filename', 'неизвестно')}")
                # Проверяем размер полученных данных
                model_data = data.get('model_data', '')
                print(f"Размер данных модели: {len(model_data) / 1024:.2f} КБ")
                return True
            else:
                print(f"Ошибка генерации: {data.get('error', 'неизвестная ошибка')}")
                return False
        else:
            print(f"Ошибка запроса: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при генерации модели: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Тест соединения с генератором 3D-моделей')
    parser.add_argument('--prompt', type=str, default='red sphere with radius 1', 
                       help='Промпт для тестовой генерации модели')
    parser.add_argument('--check-only', action='store_true', 
                       help='Только проверить соединение без генерации модели')
    
    args = parser.parse_args()
    
    print("=== Тестирование соединения с генератором 3D-моделей ===")
    print(f"Настройки подключения:")
    print(f"  Health check URL: {HEALTH_CHECK_URL}")
    print(f"  Generation API URL: {GENERATION_API_URL}")
    print(f"  Timeout: {REQUEST_TIMEOUT} секунд")
    print("=" * 50)
    
    if check_generator_status():
        print("\n✅ Соединение с генератором установлено успешно!")
        
        if not args.check_only:
            print("\n=== Тестирование генерации модели ===")
            if test_model_generation(args.prompt):
                print("\n✅ Тест генерации пройден успешно!")
            else:
                print("\n❌ Тест генерации не пройден.")
    else:
        print("\n❌ Невозможно установить соединение с генератором.")
        print("Проверьте следующее:")
        print("  1. Запущен ли скрипт shape_server.py на вашем локальном ПК?")
        print("  2. Правильно ли указан IP-адрес в файле config.py?")
        print("  3. Открыт ли порт на вашем ПК и доступен ли он из VDS?")
        print("  4. Нет ли проблем с сетевым подключением или брандмауэром?")

if __name__ == "__main__":
    main() 