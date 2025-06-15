#!/usr/bin/env python3
# Скрипт для тестирования функции проверки статуса генератора
from remote_generator import check_generator_status

print("Тестирование функции проверки статуса генератора...")
result = check_generator_status()
print(f"Результат: {result}")

# Проверка успешности результата
if isinstance(result, dict) and result.get('status') == 'online':
    print("ТЕСТ ПРОЙДЕН: Генератор доступен!")
else:
    print("ТЕСТ НЕ ПРОЙДЕН: Генератор недоступен или возникла ошибка.") 