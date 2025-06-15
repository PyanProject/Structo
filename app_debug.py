#!/usr/bin/env python3
# Тестовый скрипт для отладки и проверки маршрута /check_generator_status

from flask import Flask, jsonify
from remote_generator import check_generator_status
import json

app = Flask(__name__)

@app.route('/test_status', methods=['GET'])
def test_status():
    """Тестовый маршрут для проверки статуса генератора"""
    try:
        print("==== Получен запрос на проверку статуса генератора ====")
        status_result = check_generator_status()
        
        print(f"==== Тип результата: {type(status_result)} ====")
        print(f"==== Значение: {status_result} ====")
        
        # Проверяем, вернулся ли кортеж с кодом ответа
        if isinstance(status_result, tuple) and len(status_result) > 1:
            status, code = status_result
            print(f"==== Статус генератора с ошибкой: {status}, код: {code} ====")
            print(f"==== Тип status: {type(status)} ====")
            result = jsonify(status)
            print(f"==== Ответ: {result.get_data(as_text=True)} ====")
            return result, code
        
        # Иначе получен обычный статус (словарь) без ошибки
        print(f"==== Статус генератора без ошибки: {status_result} ====")
        try:
            # Пробуем обработать результат как словарь вручную
            if isinstance(status_result, dict):
                json_result = json.dumps(status_result)
                print(f"==== Закодированный JSON вручную: {json_result} ====")
            
            # Проверим работу jsonify
            result = jsonify(status_result)
            print(f"==== Ответ jsonify: {result.get_data(as_text=True)} ====")
            return result
        except Exception as e:
            print(f"==== Ошибка при сериализации JSON: {str(e)} ====")
            # Возвращаем сырые данные для отладки
            return str(status_result), 500
    except Exception as e:
        print(f"==== Ошибка при проверке статуса генератора: {str(e)} ====")
        return jsonify({
            'status': 'error',
            'message': f'Ошибка сервера при проверке статуса: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5050, host='0.0.0.0') 