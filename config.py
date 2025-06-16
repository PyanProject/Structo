LOCAL_PC_IP = "176.192.205.93"  # Проверьте актуальность айпишника!

# Порт на котором работает API
LOCAL_PC_PORT = 5678

# Чек API
GENERATION_API_URL = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/generate"
HEALTH_CHECK_URL = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/health"
PROGRESS_CHECK_URL = f"http://{LOCAL_PC_IP}:{LOCAL_PC_PORT}/progress"

# Таймаут для запросов 
REQUEST_TIMEOUT = 300  

# Максимальный размер получаемого файла
MAX_MODEL_SIZE = 50 * 1024 * 1024  # 50 МБ 