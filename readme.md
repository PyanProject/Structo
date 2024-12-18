# **RU**

ОПИСАНИЕ
**Modelit** — это веб-приложение для генерации 3D-моделей на основе текстовых описаний. Пользователь может описать желаемую модель, указывая цвет, форму и размер, и система сгенерирует соответствующую 3D-модель. Приложение использует генеративно-состязательные сети (GAN), а также предварительно обученные модели, такие как CLIP, для преобразования текста в эмбеддинги.

# Особенности
- Генерация 3D-моделей из текстовых описаний.
- Поддержка заданных форм и цветов:
  - **Формы**: Сфера, куб, конус, цилиндр, пирамида, тор, цилиндрическая труба.
  - **Цвета**: Красный, синий, зелёный, жёлтый, фиолетовый, оранжевый, белый.
- Визуализация 3D-моделей прямо в браузере.
- Сохранение и загрузка сгенерированных моделей в формате `.ply`.

# Технологии
- **Python Flask**: Для серверной части.
- **PyTorch**: Для работы с нейросетями и GAN.
- **trimesh**: Для работы с 3D-моделями.
- **Three.js**: Для визуализации 3D-объектов в браузере.
- **HTML/CSS/JavaScript**: Для пользовательского интерфейса.

# Установка

1. **Клонирование репозитория**
   ```bash
   git clone <URL>
   cd Modelit
   ```

2. **Установка зависимостей**
   Убедитесь, что Python 3.8+ установлен, затем выполните:
   ```bash
   pip install -r requirements.txt
   ```

3. **Настройка**
   - Создайте папку `models` для хранения сгенерированных файлов.
   - Поместите предварительно обученные веса GAN (файлы `generator.pth` и `discriminator.pth`) в корень проекта.

4. **Запуск**
   ```bash
   python app.py
   ```
   Сервер будет доступен по адресу: `http://127.0.0.1:5000`.

# Использование
1. Откройте приложение в браузере.
2. Введите описание модели в текстовое поле (например, "Красная сфера с радиусом 1").
3. Нажмите кнопку "Сгенерировать".
4. После завершения генерации вы увидите визуализацию модели. Нажмите кнопку загрузки, чтобы сохранить `.ply` файл.

# Структура проекта
- **`app.py`**: Основной сервер приложения.
- **`model_generator.py`**: Логика генерации 3D-моделей.
- **`gan_model.py`**: Определение генератора и дискриминатора GAN.
- **`embedding_generator.py`**: Генерация текстовых эмбеддингов с помощью CLIP.
- **`static/`**: Стили и скрипты для интерфейса.
- **`templates/`**: HTML-шаблоны.

# Пример текста для генерации
- "Красный конус с высотой 2".
- "Синяя сфера радиусом 1.5".

# Возможные улучшения
- Добавление большего количества форм и цветов.
- Интеграция генерации текстур для 3D-моделей.
- Поддержка генерации более сложных композиций.

# Авторство
© 2024 Pyan Inc.  
Для вопросов и предложений свяжитесь с нами по адресу: `alvttttttt@gmail.com`.


# Лицензия
MIT License

# **ENG**

# Description
**Modelit** is a web application for generating 3D models from text descriptions. Users can specify the desired model's color, shape, and size, and the system will generate a corresponding 3D model. The application leverages Generative Adversarial Networks (GANs) and pre-trained models like CLIP to transform text into embeddings.

# Features
- Generate 3D models from text descriptions.
- Support for predefined shapes and colors:
  - **Shapes**: Sphere, cube, cone, cylinder, pyramid, torus, cylindrical tube.
  - **Colors**: Red, blue, green, yellow, violet, orange, white.
- Visualize 3D models directly in the browser.
- Save and download generated models in `.ply` format.

# Technologies
- **Python Flask**: For the backend.
- **PyTorch**: For neural networks and GAN operations.
- **trimesh**: For handling 3D models.
- **Three.js**: For rendering 3D objects in the browser.
- **HTML/CSS/JavaScript**: For the user interface.

# Installation

1. **Clone the repository**
   ```bash
   git clone <URL>
   cd Modelit
   ```

2. **Install dependencies**
   Make sure Python 3.8+ is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup**
   - Create a `models` folder to store generated files.
   - Place pre-trained GAN weights (`generator.pth` and `discriminator.pth`) in the root directory.

4. **Run the application**
   ```bash
   python app.py
   ```
   The server will be available at `http://127.0.0.1:5000`.

# Usage
1. Open the application in your browser.
2. Enter a description of the model in the input field (e.g., "A red sphere with a radius of 1").
3. Click the "Generate" button.
4. After the model is generated, you'll see its visualization. Click the download button to save the `.ply` file.

# Project Structure
- **`app.py`**: Main application server.
- **`model_generator.py`**: Logic for generating 3D models.
- **`gan_model.py`**: Definitions of the GAN generator and discriminator.
- **`embedding_generator.py`**: Generating text embeddings with CLIP.
- **`static/`**: Styles and scripts for the interface.
- **`templates/`**: HTML templates.

# Sample Input for Generation
- "A red cone with height 2".
- "A blue sphere with a radius of 1.5".

# Potential Improvements
- Add more shapes and colors.
- Integrate texture generation for 3D models.
- Support the generation of more complex compositions.

# Author
© 2024 Pyan Inc.  
For questions and suggestions, contact us at: `alvttttttt@gmai.com`.
