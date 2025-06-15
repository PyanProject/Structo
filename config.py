# Конфигурация для соединения с сервером генерации на локальном ПК

# IP-адрес вашего ПК в сети
# Проверьте свой текущий IP-адрес и обновите здесь, если он изменился
LOCAL_PC_IP = "176.192.205.93"  # Проверьте актуальность этого IP-адреса!

# Порт, на котором работает API-сервер
LOCAL_PC_PORT = 5678

# Полные URL для API
GENERATION_API_URL = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/generate"
HEALTH_CHECK_URL = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/health"
PROGRESS_CHECK_URL = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/progress"

# Таймаут для запросов (в секундах)
REQUEST_TIMEOUT = 300  # 5 минут для генерации модели

# Максимальный размер получаемого файла (в байтах)
MAX_MODEL_SIZE = 50 * 1024 * 1024  # 50 МБ 