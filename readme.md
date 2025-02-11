### README.md

# ModelIT

## 📌 Описание проекта
Этот репозиторий содержит основные компоненты проекта ModelIT, включая эксперименты, основную модель и обученные веса.

### 📂 Структура проекта
- `experiments/` — новые идеи и реализации, экспериментальные пункты.
- `main/` — основная модель с новой категоризацией.
- `model-pth's/` — обученные веса (актуальные).

---

## 🚀 Установка и настройка окружения

### 1️⃣ Создание виртуального окружения
Перед установкой зависимостей необходимо создать и активировать виртуальное окружение:

#### 🔹 Windows (cmd)
```bash
cd D:\gitpub\modelit
python -m venv venv
venv\Scripts\activate
```

#### 🔹 Windows (PowerShell)
```powershell
cd D:\gitpub\modelit
python -m venv venv
venv\Scripts\Activate.ps1
```
⚠ Если выдаёт ошибку "скрипты запрещены", выполните:
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```

#### 🔹 macOS / Linux
```bash
cd /path/to/modelit
python3 -m venv venv
source venv/bin/activate
```

---

### 2️⃣ Установка зависимостей
После активации виртуального окружения установите зависимости:
```bash
pip install -r requirements.txt
```

Если необходимо обновить пакеты:
```bash
pip install --upgrade -r requirements.txt
```

Для установки без кэша:
```bash
pip install --no-cache-dir -r requirements.txt
```

---

## 📖 Дополнительная информация
Более подробные инструкции по запуску находятся в [MAIN.md](./MAIN.md).
