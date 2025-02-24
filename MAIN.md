---

### MAIN.md

# Руководство по работе с проектом ModelIT

Ниже приведены инструкции, как подготовить окружение и запустить различные компоненты проекта.

---

## 1️⃣ Подготовка окружения

### 🔹 Структура проекта
Убедитесь, что у вас есть следующие файлы и каталоги:

```
project_root/
├── experiments/
├── main/
├── model-pth's/
├── scripts/
├── datasets/
├── utils/
├── logs/
├── requirements.txt
├── run.py
└── README.md
```

### 🔹 Установка зависимостей
```bash
pip install -r requirements.txt
```

Если используются модели spaCy, установите их отдельно:
```bash
python -m spacy download ru_core_news_sm
python -m spacy download en_core_web_sm
```

---

## 2️⃣ Запуск веб-приложения
Если проект включает веб-интерфейс (например, Flask), запустите сервер:
```bash
python run.py
```
После этого приложение будет доступно по адресу [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

---

## 3️⃣ Запуск обучения модели
Для обучения модели используйте скрипт `scripts/train_gan.py`. Он загружает датасет, запускает обучение и сохраняет метрики.

Запуск:
```bash
python scripts/train_gan.py
```

---

## 4️⃣ Генерация 3D-модели по тексту
Если в проекте есть генерация 3D-моделей, используйте:
```bash
python scripts/generate_model.py
```

---

## 5️⃣ Поиск 3D-моделей по тексту
```bash
python scripts/retrieval.py
```

---

## 🔄 Итоговый порядок запуска

1. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   python -m spacy download ru_core_news_sm
   python -m spacy download en_core_web_sm
   ```

2. Запустить веб-сервер:
   ```bash
   python run.py
   ```

3. Запустить обучение модели:
   ```bash
   python scripts/train_gan.py
   ```

4. Сгенерировать 3D-модель:
   ```bash
   python scripts/generate_model.py
   ```

5. Найти 3D-модель по тексту:
   ```bash
   python scripts/retrieval.py
   ```

Все логи и результаты сохраняются в `logs/`.