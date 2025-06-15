from flask import Flask

app = Flask(__name__)

@app.route('/check_generator_status')
def check_status():
    """Простой тестовый маршрут, возвращающий JSON"""
    response_json = '{"status": "online", "gpu_available": true}'
    return response_json, 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 