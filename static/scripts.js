// Глобальные переменные
let scene, camera, renderer, controls, loadedModel;
let isModelLoaded = false;
let generationInProgress = false;
let modelPollingInterval = null; // Для периодических проверок статуса модели
let progressInterval = null; // Для анимации прогресс-бара
let lastGeneratedModelUrl = null; // Хранит URL последней сгенерированной модели
let lastDownloadUrl = null; // Хранит URL для скачивания последней модели

// Идеи для промптов
globallang = "";

const promptIdeas = [
    "Красная сфера с радиусом 1",
    "Голубой куб со стороной 2",
    "Зеленый цилиндр высотой 3 и радиусом 1",
    "Желтая пирамида с основанием 2x2",
    "Фиолетовый тор с внешним радиусом 2 и внутренним 1",
    "Оранжевый конус высотой 3 и радиусом основания 1.5",
    "Радужный додекаэдр с ребром 1",
    "Серебряный икосаэдр с ребром 1",
    "Золотой октаэдр с ребром 1",
    "Кристалл изумруда с огранкой",
    "Деревянная чаша с диаметром 10",
    "Кулон в форме сердца",
    "Ваза с текстурой мрамора",
    "Кольцо с драгоценным камнем",
    "Настольная лампа в стиле арт-деко",
    "Модель человеческого черепа",
    "Фигурка шахматного коня",
    "Модель автомобиля Tesla Cybertruck",
    "Горный пейзаж с озером",
    "Бюст Аристотеля"
];

// Событие загрузки документа
document.addEventListener('DOMContentLoaded', function() {
    // Инициализация сцены Three.js
    initScene();
    
    // Добавляем обработчики событий
    setupEventListeners();
    
    // Проверяем флаг автоматического обновления после завершения генерации
    const generationCompleted = localStorage.getItem('generationCompleted');
    if (generationCompleted === 'true') {
        console.log('Обнаружен флаг завершенной генерации. Запрашиваем загрузку модели...');
        
        // Показываем индикатор загрузки
        const progressContainer = document.getElementById('progress-container');
        const progressText = document.getElementById('progress-text');
        if (progressContainer && progressText) {
            progressContainer.style.display = 'block';
            progressText.textContent = 'Загрузка сгенерированной модели...';
        }
        
        // Сначала удаляем флаг, чтобы избежать зацикливания
        localStorage.removeItem('generationCompleted');
        
        // Запрашиваем текущий статус модели
        fetch('/generation_progress')
            .then(response => response.json())
            .then(data => {
                console.log('Получен статус модели после перезагрузки:', data);
                
                if (data && data.status === 'completed') {
                    // Если есть filename, но нет file_ready, запускаем скачивание модели
                    if (data.filename && !data.file_ready) {
                        console.log('Модель сгенерирована, но не загружена на VDS. Инициируем загрузку...');
                        const downloadUrl = `/proxy/download/${data.filename}`;
                        
                        fetch(downloadUrl)
                            .then(response => response.json())
                            .then(downloadData => {
                                if (downloadData.status === 'success') {
                                    console.log('Модель успешно загружена:', downloadData.vds_model_url);
                                    
                                    // Сохраняем URL модели и загружаем ее
                                    localStorage.setItem('lastModelUrl', downloadData.vds_model_url);
                                    const refreshedUrl = downloadData.vds_model_url + '?t=' + new Date().getTime();
                                    loadModelWithRetry(refreshedUrl);
                                    
                                    // Настраиваем кнопку скачивания
                                    if (downloadData.vds_download_url) {
                                        localStorage.setItem('lastDownloadUrl', downloadData.vds_download_url);
                                        const downloadBtn = document.getElementById('download-button');
                                        
                                        if (downloadBtn) downloadBtn.style.display = 'flex';
                                    }
                                    
                                    // Скрываем прогресс
                                    if (progressContainer) {
                                        progressContainer.style.display = 'none';
                                    }
                                } else {
                                    console.error('Ошибка при загрузке модели:', downloadData.message);
                                    showError('Ошибка при загрузке модели: ' + downloadData.message);
                                }
                            })
                            .catch(error => {
                                console.error('Ошибка при запросе загрузки:', error);
                                showError('Ошибка при загрузке модели');
                            });
                    } 
                    // Если модель уже доступна на VDS
                    else if (data.vds_model_url) {
                        console.log('Модель уже доступна на VDS:', data.vds_model_url);
                        
                        // Загружаем модель напрямую
                        localStorage.setItem('lastModelUrl', data.vds_model_url);
                        const refreshedUrl = data.vds_model_url + '?t=' + new Date().getTime();
                        loadModelWithRetry(refreshedUrl);
                        
                        // Настраиваем кнопку скачивания
                        if (data.vds_download_url) {
                            localStorage.setItem('lastDownloadUrl', data.vds_download_url);
                            const downloadBtn = document.getElementById('download-button');
                            
                            if (downloadBtn) downloadBtn.style.display = 'flex';
                        }
                        
                        // Скрываем прогресс
                        if (progressContainer) {
                            progressContainer.style.display = 'none';
                        }
                    }
                }
            })
            .catch(error => {
                console.error('Ошибка при получении статуса модели:', error);
                showError('Ошибка при проверке статуса модели');
            });
    }
    
    // Проверяем доступность локального сервера генерации
    checkLocalServerAvailability();
    
    // Проверяем наличие предыдущей генерации
    checkForPreviousGeneration();
    
    // Показываем случайную идею для промпта
    showRandomIdea();

    // Функция для получения текущего языка при загрузке страницы
    getCurrentLanguage();

    const loader = document.getElementById('loader-overlay');
    if (loader) {
        // Принудительно скрываем loader через 3.5 секунды
        setTimeout(() => {
            loader.style.display = 'none';
        }, 3700);
    }

    const accountLink = document.getElementById('account-link');
    if (accountLink) {
        let hoverTimer = null;
        accountLink.addEventListener('mouseenter', function() {
            if (accountLink.getAttribute('data-authenticated') === 'true') {
                hoverTimer = setTimeout(() => {
                    if (typeof accountLink.onclick === 'function') {
                        accountLink.onclick();
                    }
                }, 200);
            }
        });
        accountLink.addEventListener('mouseleave', function(event) {
            if (hoverTimer) {
                clearTimeout(hoverTimer);
                hoverTimer = null;
            }
            // Проверяем, ушёл ли курсор на user-menu
            const userMenu = document.getElementById('user-menu');
            let toElement = event.relatedTarget || event.toElement;
            if (userMenu && userMenu.style.display === 'block') {
                if (userMenu.contains(toElement)) {
                    // Курсор ушёл на меню — не закрываем
                    return;
                }
                if (typeof accountLink.onclick === 'function') {
                    accountLink.onclick();
                }
            }
        });
        // Добавляем обработчик mouseleave для user-menu (только один раз)
        if (!userMenu.hasMouseleaveHandler) {
            userMenu.addEventListener('mouseleave', function(event) {
                let toElement = event.relatedTarget || event.toElement;
                if (accountLink.contains(toElement)) {
                    return;
                }
                if (userMenu.style.display === 'block') {
                    if (typeof accountLink.onclick === 'function') {
                        accountLink.onclick();
                    }
                }
            });
            userMenu.hasMouseleaveHandler = true;
        }
    }

    // Обработчик для кнопки 'Подписки' на главной (теперь это button)
    const subsBtn = document.getElementById('subs-link-btn');
    if (subsBtn) {
        subsBtn.addEventListener('click', function(e) {
            console.log('Клик по кнопке Подписки');
            handleSubscriptionsClick();
        });
    }
    
});

// После загрузки DOM проверяем возможную генерацию
document.addEventListener('DOMContentLoaded', async () => {
    // Показываем анимацию загрузки
    const loaderOverlay = document.getElementById('loader-overlay');
    if (loaderOverlay) {
        loaderOverlay.style.display = 'flex';
    }
    
    const pageReloaded = performance.navigation.type === 1;
    const remembered = sessionStorage.getItem('remembered') === 'true';

    // Проверяем статус авторизации на сервере
    try {
        const response = await fetch('/auth_status');
        const data = await response.json();

        const accountLink = document.getElementById('account-link');
        if (data.authenticated) {
            console.log('[DEBUG] Пользователь авторизован на сервере.');
            updateUserUI(data.username);
            accountLink.onclick = toggleUserMenu;
        } else {
            console.log('[DEBUG] Пользователь не авторизован на сервере.');
            accountLink.textContent = 'Аккаунт';
            accountLink.onclick = openModal;
        }
    } catch (error) {
        console.error('Ошибка проверки авторизации:', error);
    }
    
    // Проверяем текущий статус генерации (особенно важно при перезагрузке страницы)
    try {
        const response = await fetch('/generation_progress');
        if (response.ok) {
            const data = await response.json();
            console.log('[DEBUG] Проверка статуса генерации при загрузке страницы:', data);
            
            // Если генерация завершена, но файл еще не загружен, инициируем автоматическое обновление
            if (data.status === 'completed' && !data.file_ready && data.filename) {
                console.log('[DEBUG] Генерация завершена, но модель не загружена. Запускаем автоматическую загрузку...');
                
                // Устанавливаем флаг и обновляем страницу через 1.5 секунды
                localStorage.setItem('generationCompleted', 'true');
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            }
        }
    } catch (error) {
        console.error('Ошибка проверки статуса генерации:', error);
    }
    
    // Скрываем анимацию загрузки через 3 секунды
    setTimeout(() => {
        if (loaderOverlay) {
            loaderOverlay.style.opacity = '0';
            setTimeout(() => {
                loaderOverlay.style.display = 'none';
            }, 500);
        }
    }, 3000);
});

// Функция для отображения случайной идеи промпта
function showRandomIdea() {
    const ideas = [
        "Красная сфера с радиусом 1",
        "Синий куб со стороной 2",
        "Кулон в виде сердца",
        "Кольцо с украшением",
        "Зеленая ваза с узором",
        "Игрушечная машинка",
        "Кресло в стиле минимализм",
        "Настольная лампа",
        "Робот с антенной на голове",
        "Ракета с соплами"
    ];
    
    const randomIndex = Math.floor(Math.random() * ideas.length);
    document.getElementById('idea-text').textContent = ideas[randomIndex];
}

// Инициализация 3D сцены
function initScene() {
    const resultBox = document.getElementById('result-box');
    
    // Создаем сцену
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x252525);
    
    // Создаем камеру
    camera = new THREE.PerspectiveCamera(75, resultBox.clientWidth / resultBox.clientHeight, 0.1, 1000);
    camera.position.z = 5;
    
    // Создаем рендерер
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(resultBox.clientWidth, resultBox.clientHeight);
    resultBox.appendChild(renderer.domElement);
    
    // Добавляем управление камерой
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Добавляем освещение
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Начинаем рендеринг
    animate();
    
    // Обработка изменения размера окна
    window.addEventListener('resize', onWindowResize, false);
}

// Анимация сцены
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Обработка изменения размера окна
function onWindowResize() {
    const resultBox = document.getElementById('result-box');
    camera.aspect = resultBox.clientWidth / resultBox.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(resultBox.clientWidth, resultBox.clientHeight);
}

// Настройка обработчиков событий
function setupEventListeners() {
    // Обработчик кнопки генерации
    const generateButton = document.getElementById('generate-button');
    const promptInput = document.getElementById('search-input');

    

    if (generateButton) {
        generateButton.addEventListener('click', async function() {
            const prompt = promptInput.value.trim();
            await generateModel(prompt);
        });
    }
    
    // Обработка нажатия Enter в поле ввода
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', async function(e) {
            if (e.key === 'Enter') {
                const prompt = searchInput.value.trim();
                await generateModel(prompt);
            }
        });
    }
    
    // Обработчик клика по идее для копирования в input
    const ideaText = document.getElementById('idea-text');
    if (ideaText) {
        ideaText.addEventListener('click', async function() {
            const ideaText = promptIdeas[Math.floor(Math.random() * promptIdeas.length)];
            promptInput.value = ideaText;
            await generateModel(ideaText);
        });
    }
}

// Имитация прогресса генерации модели
function startProgressSimulation(autoComplete = true) {
    console.log('Запуск имитации прогресса генерации');
    
    // Показываем контейнер с прогрессом
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    progressContainer.style.display = 'block';
    progressBar.style.width = '5%'; // Начальный прогресс
    progressText.textContent = 'Отправка запроса...';
    
    // Если уже есть интервал, очищаем его
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    let progress = 5;
    
    // Запускаем интервал для обновления прогресса
    progressInterval = setInterval(() => {
        // Обновляем текст в зависимости от стадии
        if (progress < 20) {
            progressText.textContent = 'Отправка запроса...';
        } else if (progress < 40) {
            progressText.textContent = 'Загрузка моделей...';
        } else if (progress < 60) {
            progressText.textContent = 'Обработка промпта...';
        } else if (progress < 80) {
            progressText.textContent = 'Генерация модели...';
        } else if (progress < 95) {
            progressText.textContent = 'Обработка результатов...';
        } else {
            progressText.textContent = 'Подготовка к отображению...';
        }
        
        // Если включено автоматическое завершение
        if (autoComplete) {
            progress += Math.random() * 3 + 1; // Случайное увеличение
            
            // Ограничиваем прогресс до 99% при автозавершении
            if (progress > 99) {
                progress = 99;
            }
        } else {
            // Для бесконечного прогресса используем циклическую анимацию
            progress += 0.5;
            if (progress >= 90) {
                progress = 60; // Сбрасываем назад для циклической анимации
            }
        }
        
        // Обновляем ширину прогресс-бара
        progressBar.style.width = progress + '%';
    }, 200);
}

// Остановка имитации прогресса
function stopProgressSimulation(success) {
    console.log('Остановка имитации прогресса, успех:', success);
    
    // Если имитация не запущена, ничего не делаем
    if (!progressInterval) return;
    
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressContainer = document.getElementById('progress-container');
    
    // Завершаем прогресс успешно или с ошибкой
    if (success) {
        progressBar.style.width = '100%';
        progressText.textContent = 'Готово!';
        
        // Через 1 секунду скрываем прогресс
        setTimeout(() => {
            progressContainer.style.display = 'none';
        }, 1000);
    } else {
        progressBar.style.backgroundColor = '#ff5768'; // Красный цвет для ошибки
        progressText.textContent = 'Ошибка генерации!';
        
        // Скрываем прогресс сразу после отображения ошибки
        progressContainer.style.display = 'none';
    }
    
    // Очищаем интервал
    clearInterval(progressInterval);
    progressInterval = null;
}

// Функция для проверки и восстановления предыдущей генерации
function checkForPreviousGeneration() {
    const lastPrompt = localStorage.getItem('lastPrompt');
    const lastModelUrl = localStorage.getItem('lastModelUrl');
    const lastDownloadUrl = localStorage.getItem('lastDownloadUrl');
    const generationStartTime = localStorage.getItem('generationStartTime');
    const generationPhase = localStorage.getItem('generationPhase');
    
    // Также проверим, есть ли запущенная генерация
    fetch('/generation_progress')
        .then(response => response.json())
        .then(data => {
            console.log('Получен статус генерации:', data);
            
            // Если генерация завершена или в процессе
            if (data && (data.status === 'completed' || data.status === 'generating')) {
                // Если статус 'completed', но модель не загружена, автоматически обновляем страницу
                if (data.status === 'completed' && data.filename && !data.file_ready) {
                    console.log('Обнаружена завершенная генерация, но модель не загружена на VDS. Запускаем автоматическую загрузку...');
                    
                    // Показываем индикатор загрузки
                    const progressContainer = document.getElementById('progress-container');
                    const progressText = document.getElementById('progress-text');
                    if (progressContainer && progressText) {
                        progressContainer.style.display = 'block';
                        progressText.textContent = 'Загрузка сгенерированной модели...';
                    }
                    
                    // Запускаем процесс скачивания модели на VDS
                    const downloadUrl = `/proxy/download/${data.filename}`;
                    console.log('Отправляем запрос на скачивание модели:', downloadUrl);
                    
                    fetch(downloadUrl)
                        .then(response => response.json())
                        .then(downloadData => {
                            if (downloadData.status === 'success') {
                                console.log('Модель успешно скачана на VDS:', downloadData.vds_model_url);
                                
                                // Загружаем модель
                                const modelUrl = downloadData.vds_model_url + '?t=' + new Date().getTime();
                                loadModelWithRetry(modelUrl);
                                
                                // Сохраняем URL модели и URL для скачивания
                                localStorage.setItem('lastModelUrl', downloadData.vds_model_url);
                                localStorage.setItem('lastDownloadUrl', downloadData.vds_download_url);
                                
                                // Настраиваем кнопки скачивания
                                const downloadBtn = document.getElementById('download-button');
                                if (downloadBtn) downloadBtn.style.display = 'flex';
                            } else {
                                console.error('Ошибка при скачивании модели на VDS:', downloadData.message);
                                showError('Ошибка при скачивании модели: ' + downloadData.message);
                            }
                        })
                        .catch(error => {
                            console.error('Ошибка при запросе скачивания:', error);
                            showError('Ошибка при запросе скачивания модели');
                        });
                }
                // Если генерация в процессе, отображаем прогресс
                else if (data.status === 'generating') {
                    console.log('Обнаружена активная генерация. Отображаем прогресс...');
                    
                    // Устанавливаем флаг, что генерация идет
                    generationInProgress = true;
                    
                    // Отображаем информацию о прогрессе
                    displayGenerationPhase('generating_3d');
                    
                    // Запускаем обновление каждую секунду
                    if (!modelPollingInterval) {
                        modelPollingInterval = setInterval(checkGenerationProgress, 1000);
                    }
                }
            }
        })
        .catch(error => {
            console.error('Ошибка при проверке статуса генерации:', error);
        });
    
    // Если есть сохраненная информация о предыдущей модели, восстанавливаем ее
    if (lastPrompt && lastModelUrl && generationStartTime) {
        console.log('Обнаружена предыдущая генерация модели:', lastPrompt);
        
        // Восстанавливаем глобальные переменные
        lastGeneratedModelUrl = lastModelUrl;
        
        // Если lastDownloadUrl не задан, создаем его на основе lastModelUrl
        if (!lastDownloadUrl) {
            let downloadUrl = lastModelUrl;
            if (downloadUrl.startsWith('/')) {
                downloadUrl = `http://176.195.64.215:5678${downloadUrl}`;
            }
            downloadUrl = downloadUrl.includes('?') ? 
                `${downloadUrl}&download=true` : 
                `${downloadUrl}?download=true`;
            window.lastDownloadUrl = downloadUrl;
        } else {
            window.lastDownloadUrl = lastDownloadUrl;
        }
        
        // Отображаем текущую фазу генерации если она есть, иначе считаем что генерация завершена
        displayGenerationPhase(generationPhase || 'completed');
        
        // Преобразуем URL модели, если он относительный
        let modelUrl = lastModelUrl;
        if (modelUrl.startsWith('/')) {
            modelUrl = `http://176.195.64.215:5678${modelUrl}`;
            console.log('Преобразован URL модели:', modelUrl);
        }
        
        // Загружаем модель
        loadModelWithRetry(modelUrl + '?t=' + new Date().getTime());
        
        // Настраиваем кнопки скачивания
        const downloadBtn = document.getElementById('download-button');
        if (downloadBtn) downloadBtn.style.display = 'flex';
    }
}

// Функция для обновления прогресс-бара
function updateProgressBar(phase, message) {
    const progressContainer = document.getElementById('progress-container');
    const progressText = document.getElementById('progress-text');
    
    // Показываем контейнер с прогрессом
    progressContainer.style.display = 'block';
    
    // Обновляем сообщение в зависимости от фазы
    if (message) {
        progressText.textContent = message;
    } else if (phase) {
        switch(phase) {
            case 'loading':
                progressText.textContent = 'Загрузка моделей...';
                break;
            case 'generating':
                progressText.textContent = 'Генерация 3D объекта...';
                break;
            case 'finalizing':
                progressText.textContent = 'Финальная обработка...';
                break;
            case 'completed':
                progressText.textContent = 'Готово!';
                // Скрываем прогресс-контейнер через 2 секунды после завершения
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 2000);
                break;
        }
    }
}

// Функция для отображения состояния генерации
function displayGenerationPhase(phase) {
    // Получаем информационный элемент
    const infoElement = document.getElementById('info-message');
    if (!infoElement) return;
    
    // Определяем содержимое в зависимости от фазы
    let title, message;
    
    switch(phase) {
        case 'loading_models':
            title = 'Загрузка моделей на локальном ПК';
            message = 'Подготовка к созданию 3D модели. Загружаются ML модели на локальном компьютере.';
            updateProgressBar('loading', 'Загрузка моделей на локальном ПК...');
            break;
            
        case 'generating_3d':
            title = 'Генерация 3D объекта';
            message = 'Создание трехмерной модели на основе текстового описания. Процесс может занять 10-20 минут.';
            updateProgressBar('generating', 'Создание 3D объекта...');
            break;
            
        case 'finalizing':
            title = 'Финальная обработка модели';
            message = 'Оптимизация и подготовка 3D модели к экспорту. Это займет еще несколько минут.';
            updateProgressBar('finalizing', 'Финальная обработка...');
            break;
            
        case 'completed':
            title = 'Модель успешно сгенерирована!';
            message = 'Модель готова и загружена в браузер.';
            updateProgressBar('completed', 'Готово!');
            break;
            
        default:
            title = 'Обработка запроса';
            message = 'Пожалуйста, подождите. Выполняется обработка запроса.';
            updateProgressBar(null, 'Инициализация...');
    }
    
    // Сохраняем текущую фазу в localStorage
    localStorage.setItem('generationPhase', phase);
    
    // Обновляем информационное сообщение
    infoElement.innerHTML = `
        <strong style="font-size: 18px; color: #6B5CE7;">${title}</strong><br>
        <span style="font-size: 16px;">${message}</span><br>
        <div style="margin-top: 10px; font-weight: bold;">
            ${phase !== 'completed' ? 'Используйте кнопку обновления, когда генерация завершится.' : 'Генерация успешно завершена!'}
        </div>
    `;
    infoElement.style.display = 'block';
    
    // Если фаза завершена, скроем сообщение через 5 секунд
    if (phase === 'completed') {
        setTimeout(() => {
            infoElement.style.display = 'none';
        }, 5000);
    }
}

// Функция для генерации модели
async function generateModel(prompt) {
    console.log('Начало генерации модели с промптом:', prompt);
    
    // Объявляем переменную subscription в начале функции
    let subscription = 1; // Значение по умолчанию
    
    // Проверяем подписку пользователя
    try {
        const response = await fetch('/check_subscription');
        const data = await response.json();
        console.log('Ответ сервера о подписке:', data); // Отладочный вывод
        subscription = data.subscription || 1; // Используем subscription_level или значение по умолчанию
        console.log('Установленный уровень подписки:', subscription);
    } catch (error) {
        console.error('Ошибка при проверке подписки:', error);
        return;
    }

    
    // Если поле ввода пустое, показываем ошибку
    if (!prompt || prompt.trim() === '') {
        console.log('Ошибка: пустой промпт');
        showError('Пожалуйста, введите промпт для генерации модели');
        return;
    }

    // Переводим промпт на английский
    try {
        console.log('Отправка промпта на перевод:', prompt);
        const translatedPrompt = await translateToEnglish(prompt);
        console.log('Переведенный промпт:', translatedPrompt);
        prompt = translatedPrompt; // Используем переведенный промпт
    } catch (error) {
        console.error('Ошибка при переводе промпта:', error);
        showError('Ошибка при переводе промпта. Попробуйте использовать английский язык.');
        return;
    }
    
    // Очищаем предыдущие сообщения об ошибках и скрываем кнопку скачивания
    const errorMessage = document.getElementById('error-message');
    const downloadButton = document.getElementById('download-button');
    
    errorMessage.style.display = 'none';
    downloadButton.style.display = 'none';
    
    // Удаляем текущую модель, если она есть
    removeLoadedModel();
    
    // Показываем прогресс-бар и начинаем анимацию
    const progressContainer = document.getElementById('progress-container');
    const progressText = document.getElementById('progress-text');
    progressContainer.style.display = 'block';
    progressText.textContent = 'Генерация...';
    
    // Отключаем кнопку генерации на время запроса
    const generateButton = document.getElementById('generate-button');
    generateButton.disabled = true;
    generateButton.style.opacity = '0.5';
    
    // URL прокси-сервера на VDS
    const proxyUrl = "/proxy/generate";
    
    console.log('Отправка запроса через прокси-сервер:', proxyUrl);
    // Отправляем запрос на генерацию через прокси
    fetch(proxyUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({ prompt: prompt, subscription: subscription })
    })
    .then(response => {
        console.log('Получен ответ:', response.status);
        // Включаем кнопку генерации
        generateButton.disabled = false;
        generateButton.style.opacity = '1';
        
        return response.json().then(data => {
            console.log('Данные ответа:', data);
            if (!response.ok) {
                throw new Error(data.error || 'Неизвестная ошибка');
            }
            return data;
        });
    })
    .then(data => {
        console.log('Успешная генерация:', data);
        
        // Обработка случая, когда сервер присылает filename вместо model_url или obj_url
        if (!data.model_url && !data.obj_url && data.filename) {
            // Создаем model_url из filename
            data.model_url = `/static/models/${data.filename}`;
            console.log('Создан model_url из filename:', data.model_url);
        }
        
        // Добавляем рефреш кнопку в результат
        addRefreshButton(data.model_url || data.obj_url, true);
        
        // Сохраняем URL для скачивания
        if (data.download_url) {
            localStorage.setItem('lastDownloadUrl', data.download_url);
        } else if (data.model_url || data.obj_url) {
            // Если нет download_url, создаем его из model_url или obj_url
            let downloadUrl = data.model_url || data.obj_url;
            // Добавляем параметр download=true для указания серверу, что это скачивание
            downloadUrl = downloadUrl.includes('?') ? 
                `${downloadUrl}&download=true` : 
                `${downloadUrl}?download=true`;
            localStorage.setItem('lastDownloadUrl', downloadUrl);
        }
        
        // Сохраняем промпт для возможного восстановления
        localStorage.setItem('lastPrompt', prompt);
        
        // Сохраняем время начала генерации
        localStorage.setItem('generationStartTime', new Date().getTime());
        
        // Начинаем сразу пытаться загрузить модель
        const modelUrl = data.model_url || data.obj_url;
        if (!modelUrl) {
            console.error('Ошибка: Нет URL модели в ответе сервера', data);
            showError('Ошибка: Сервер не вернул URL модели. Проверьте логи сервера.');
            return;
        }
        
        console.log('Загрузка модели по URL:', modelUrl);
        loadModelWithRetry(modelUrl);
        
        // Запускаем мониторинг статуса генерации
        startModelMonitoring(modelUrl);
        
        // Сразу проверяем статус генерации, чтобы скрыть сообщение об ошибке
        setTimeout(() => {
            checkGenerationProgress();
        }, 1000);
    })
    .catch(error => {
        console.error('Ошибка при генерации модели:', error);
        showError('Ошибка генерации модели: ' + error.message);
        
        // Включаем кнопку генерации
        generateButton.disabled = false;
        generateButton.style.opacity = '1';
        
        // Не скрываем прогресс-контейнер при ошибке
        // Пусть индикатор продолжает крутиться
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) {
            progressContainer.style.display = 'block';
            progressText.textContent = 'Генерация...';
        }
    });
}

// Функция для мониторинга прогресса генерации модели
function startModelMonitoring(modelUrl) {
    // Сохраняем URL модели в localStorage для возможности восстановления при обновлении страницы
    localStorage.setItem('lastModelUrl', modelUrl);
    
    // URL для проверки прогресса через API сервера
    const progressCheckUrl = "/generation_progress";
    localStorage.setItem('progressCheckUrl', progressCheckUrl);
    
    // Останавливаем предыдущий мониторинг, если он был
    const prevIntervalId = parseInt(localStorage.getItem('monitoringIntervalId'));
    if (prevIntervalId) {
        clearInterval(prevIntervalId);
        localStorage.removeItem('monitoringIntervalId');
    }
    
    console.log('Запуск мониторинга генерации модели, проверка URL:', progressCheckUrl);
    
    // Проверяем состояние каждую секунду
    const intervalId = setInterval(() => {
        // Проверяем статус генерации через API
        checkGenerationProgress();
        
        // Также проверяем размер файла модели
        fetch(modelUrl + '?nocache=' + new Date().getTime(), { method: 'HEAD' })
            .then(response => {
                if (response.ok) {
                    const fileSize = parseInt(response.headers.get('Content-Length') || '0');
                    console.log('Текущий размер файла:', fileSize, 'байт');
                    
                    // Если файл достаточно большой (больше 100KB), считаем что генерация завершена
                    if (fileSize > 100000) {
                        console.log('Файл достаточного размера, загружаем модель');
                        // Принудительно скрываем прогресс-контейнер
                        const progressContainer = document.getElementById('progress-container');
                        if (progressContainer) {
                            progressContainer.style.display = 'none';
                        }
                        
                        // Загружаем модель
                        loadModelWithRetry(modelUrl + '?t=' + new Date().getTime());
                        
                        // Останавливаем мониторинг
                        clearInterval(intervalId);
                        localStorage.removeItem('monitoringIntervalId');
                    }
                }
            })
            .catch(error => console.error('Ошибка проверки файла:', error));
    }, 1000); // Интервал в 1 секунду
    
    // Сохраняем ID интервала для возможности остановки
    localStorage.setItem('monitoringIntervalId', intervalId);
    
    // Через 15 минут останавливаем мониторинг в любом случае
    setTimeout(() => {
        clearInterval(intervalId);
        localStorage.removeItem('monitoringIntervalId');
    }, 15 * 60 * 1000);
}

// Изменяем функцию checkGenerationProgress для работы через прокси
function checkGenerationProgress() {
    // Получаем URL для проверки прогресса из localStorage
    const progressCheckUrl = localStorage.getItem('progressCheckUrl') || "/proxy/progress";
    
    fetch(progressCheckUrl)
        .then(response => response.json())
        .then(data => {
            console.log('Статус генерации:', data);
            
            const progressText = document.getElementById('progress-text');
            const progressContainer = document.getElementById('progress-container');
            
            // Если получен статус завершения
            if (data && data.status === 'completed') {
                console.log('Получен статус завершения генерации');
                
                // Устанавливаем флаг для обнаружения перезагрузки страницы
                localStorage.setItem('generationCompleted', 'true');
                
                // Обновляем текст, показываем что страница сейчас обновится
                if (progressText) {
                    progressText.textContent = 'Генерация завершена! Загрузка модели...';
                }
                
                // Принудительное обновление страницы через 1.5 секунды
                console.log('Автоматическое обновление страницы через 1.5 секунды...');
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
                
                // Останавливаем мониторинг
                const intervalId = parseInt(localStorage.getItem('monitoringIntervalId'));
                if (intervalId) {
                    clearInterval(intervalId);
                    localStorage.removeItem('monitoringIntervalId');
                }
            } else if (data && data.status === 'error') {
                // Обработка ошибки
                console.error('Ошибка генерации:', data.message);
                showError(data.message || 'Ошибка при генерации модели');
                
                // Останавливаем мониторинг
                const intervalId = parseInt(localStorage.getItem('monitoringIntervalId'));
                if (intervalId) {
                    clearInterval(intervalId);
                    localStorage.removeItem('monitoringIntervalId');
                }
            } else if (data && data.status === 'generating') {
                // Обновляем сообщение о прогрессе
                if (progressText && data.message) {
                    progressText.textContent = data.message;
                }
            }
        })
        .catch(error => {
            console.error('Ошибка при проверке статуса генерации:', error);
        });
}

// Функция для надежной загрузки модели с повторными попытками
function loadModelWithRetry(modelUrl, maxRetries = 3) {
    let retries = 0;
    
    // Проверяем, определен ли URL
    if (!modelUrl) {
        console.error('Ошибка: URL модели не определен');
        showError('Ошибка: URL модели не определен');
        return;
    }
    
    // Сразу скрываем сообщение об ошибке при любой попытке загрузки модели
    const errorElement = document.getElementById('error-message');
    if (errorElement) {
        errorElement.style.display = 'none';
    }
    
    // Преобразуем URL, если это необходимо
    // Для моделей с VDS уже используется относительный путь, который не требует модификации
    if (typeof modelUrl === 'string' && modelUrl.startsWith('/output/')) {
        // Старый формат URL - заменяем на URL прокси-сервера
        // Запрашиваем скачивание файла на VDS
        const filename = modelUrl.replace('/output/', '');
        const downloadUrl = `/proxy/download/${filename}`;
        
        console.log('Запрос на скачивание модели на VDS:', downloadUrl);
        
        // Отправляем запрос на скачивание файла на VDS
        fetch(downloadUrl)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    console.log('Модель успешно скачана на VDS:', data.vds_model_url);
                    // Загружаем модель с VDS
                    modelUrl = data.vds_model_url;
                    attemptLoad();
                } else {
                    console.error('Ошибка при скачивании модели на VDS:', data.message);
                    showError('Ошибка при скачивании модели: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Ошибка при запросе скачивания:', error);
                showError('Ошибка при запросе скачивания модели');
            });
    } else if (typeof modelUrl === 'string' && modelUrl.startsWith('/static/models/')) {
        console.log('Использование модели с VDS:', modelUrl);
        attemptLoad();
    } else if (typeof modelUrl === 'string' && modelUrl.startsWith('http://176.195.64.215:5678')) {
        // Старый формат полного URL - извлекаем имя файла и заменяем на URL прокси-сервера
        const urlParts = modelUrl.split('/');
        const filename = urlParts[urlParts.length - 1].split('?')[0];
        const downloadUrl = `/proxy/download/${filename}`;
        
        console.log('Запрос на скачивание модели на VDS:', downloadUrl);
        
        // Отправляем запрос на скачивание файла на VDS
        fetch(downloadUrl)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    console.log('Модель успешно скачана на VDS:', data.vds_model_url);
                    // Загружаем модель с VDS
                    modelUrl = data.vds_model_url;
                    attemptLoad();
                } else {
                    console.error('Ошибка при скачивании модели на VDS:', data.message);
                    showError('Ошибка при скачивании модели: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Ошибка при запросе скачивания:', error);
                showError('Ошибка при запросе скачивания модели');
            });
    } else {
        console.log('Использование оригинального URL модели:', modelUrl);
        attemptLoad();
    }
    
    function attemptLoad() {
        console.log(`Попытка загрузки модели ${retries + 1}/${maxRetries + 1}`);
        console.log('URL модели для загрузки:', modelUrl);
        
        // Создаем кастомный загрузчик с поддержкой вершинных цветов
        const loader = typeof CustomOBJLoader !== 'undefined' ? new CustomOBJLoader() : new THREE.OBJLoader();
        
        // Загружаем модель
        loader.load(
            modelUrl,
            // Обработчик успешной загрузки
            (object) => {
                console.log('Модель успешно загружена');
                
                // Удаляем старую модель, если есть
                removeLoadedModel();
                
                // Устанавливаем смещение для центрирования объекта
                const box = new THREE.Box3().setFromObject(object);
                const center = box.getCenter(new THREE.Vector3());
                object.position.set(-center.x, -center.y, -center.z);
                
                // Нормализуем размер объекта
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                if (maxDim > 2) {
                    const scale = 2 / maxDim;
                    object.scale.set(scale, scale, scale);
                }
                
                // Применяем материал к объекту с учетом вершинных цветов
                object.traverse((child) => {
                    if (child instanceof THREE.Mesh) {
                        // Проверяем наличие вершинных цветов
                        if (child.geometry.hasAttribute('color')) {
                            console.log('Обнаружены вершинные цвета, применяем их');
                            const material = new THREE.MeshPhongMaterial({
                                vertexColors: true,
                                specular: 0x111111,
                                shininess: 25
                            });
                            child.material = material;
                        } else {
                            console.log('Вершинные цвета не обнаружены, применяем стандартный материал');
                            const material = new THREE.MeshPhongMaterial({
                                color: 0xaaaaaa,
                                specular: 0x111111,
                                shininess: 25
                            });
                            child.material = material;
                        }
                        
                        // Добавляем нормали, если их нет
                        if (!child.geometry.hasAttribute('normal')) {
                            child.geometry.computeVertexNormals();
                        }
                    }
                });
                
                // Добавляем объект в сцену
                scene.add(object);
                loadedModel = object;
                isModelLoaded = true;
                
                // Сбрасываем камеру
                resetCamera();
                
                // Обновляем рендер
                renderer.render(scene, camera);
                
                // Скрываем сообщение об ошибке
                const errorMessage = document.getElementById('error-message');
                if (errorMessage) {
                    errorMessage.style.display = 'none';
                }
                
                // Принудительно скрываем прогресс-контейнер
                const progressContainer = document.getElementById('progress-container');
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
                
                // Показываем кнопку скачивания
                const downloadBtn = document.getElementById('download-button');
                if (downloadBtn) {
                    downloadBtn.style.display = 'flex';
                }
                
                // Останавливаем мониторинг
                const intervalId = parseInt(localStorage.getItem('monitoringIntervalId'));
                if (intervalId) {
                    clearInterval(intervalId);
                    localStorage.removeItem('monitoringIntervalId');
                }
            },
            // Обработчик прогресса
            (xhr) => {
                console.log(`${(xhr.loaded / xhr.total * 100).toFixed(2)}% загружено`);
            },
            // Обработчик ошибок
            (error) => {
                console.error('Ошибка при загрузке модели:', error);
                
                if (retries < maxRetries) {
                    retries++;
                    console.log(`Повторная попытка ${retries}/${maxRetries}`);
                    setTimeout(attemptLoad, 3000);
                } else {
                    console.error('Превышено число попыток загрузки модели');
                    
                    // Получаем URL для проверки прогресса из localStorage
                    const progressCheckUrl = localStorage.getItem('progressCheckUrl') || "http://176.195.64.215:5678/progress";
                    
                    fetch(progressCheckUrl, {
                        headers: {
                            'Accept': 'application/json',
                            'Origin': window.location.origin
                        }
                    })
                        .then(response => response.json())
                        .then(data => {
                            if (data && data.status && data.status !== 'error' && data.status !== 'idle') {
                                console.log('Генерация идет, не показываем ошибку');
                            } else {
                                showError('Не удалось загрузить модель');
                            }
                        })
                        .catch(() => {
                            showError('Не удалось загрузить модель');
                        });
                }
            }
        );
    }
    
    // Начинаем первую попытку загрузки
    attemptLoad();
}

// Добавление кнопки обновления модели
function addRefreshButton(modelUrl, useLargeButton = false) {
    console.log('Добавление кнопки обновления для URL:', modelUrl);
    
    // Удаляем существующую кнопку, если она есть
    const existingButton = document.getElementById('refresh-button');
    if (existingButton && existingButton.parentNode) {
        existingButton.parentNode.removeChild(existingButton);
    }
    
    // Создаем новую кнопку
    const refreshButton = document.createElement('button');
    refreshButton.id = 'refresh-button';
    
    // Используем большую кнопку, если запрошено
    if (useLargeButton) {
        refreshButton.className = 'refresh-btn-large';
        refreshButton.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;">
                    <path d="M23 4v6h-6"></path>
                    <path d="M1 20v-6h6"></path>
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"></path>
                    <path d="M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                </svg>
                Обновить модель
            </div>
        `;
    } else {
        refreshButton.className = 'refresh-btn';
        refreshButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M23 4v6h-6"></path>
                <path d="M1 20v-6h6"></path>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"></path>
                <path d="M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
            </svg>
        `;
    }
    
    refreshButton.title = 'Обновить модель';
    refreshButton.style.display = 'flex'; // Гарантируем, что кнопка видима
    
    // Добавляем обработчик нажатия
    refreshButton.onclick = function() {
        console.log('Нажата кнопка обновления модели');
        
        // Показываем сообщение об обновлении
        const infoMessage = document.getElementById('info-message');
        infoMessage.textContent = 'Обновление модели...';
        infoMessage.style.display = 'block';
        
        // Удаляем текущую модель из сцены
        removeLoadedModel();
        
        // Загружаем модель с новым временным параметром
        const refreshedUrl = modelUrl.split('?')[0] + '?t=' + new Date().getTime();
        console.log('Обновление модели с URL:', refreshedUrl);
        
        // Загружаем модель с повторными попытками
        loadModelWithRetry(refreshedUrl);
        
        // Обновляем сообщение через несколько секунд
        setTimeout(() => {
            infoMessage.innerHTML = `
                <strong style="font-size: 18px; color: #6B5CE7;">Модель обновлена!</strong><br>
                Если вы все еще видите заглушку, значит генерация еще не завершена.<br>
                <div style="margin-top: 10px; font-weight: bold;">
                    Подождите еще 1-2 минуты и нажмите кнопку обновления снова.
                </div>
            `;
        }, 2000);
    };
    
    // Добавляем кнопку в контейнер с результатом
    const resultBox = document.getElementById('result-box');
    if (resultBox) {
        if (useLargeButton) {
            // Создаем контейнер для большой кнопки под 3D-вьювером
            const buttonContainer = document.createElement('div');
            buttonContainer.style.width = '100%';
            buttonContainer.style.textAlign = 'center';
            buttonContainer.style.marginTop = '15px';
            buttonContainer.appendChild(refreshButton);
            resultBox.appendChild(buttonContainer);
        } else {
            resultBox.appendChild(refreshButton);
        }
        console.log('Кнопка обновления добавлена в DOM');
    } else {
        console.error('Элемент result-box не найден!');
    }
}

// Сброс положения камеры
function resetCamera() {
    camera.position.set(0, 0, 5);
    controls.reset();
}

// Удаление загруженной модели
function removeLoadedModel() {
    if (isModelLoaded && loadedModel) {
        scene.remove(loadedModel);
        isModelLoaded = false;
    }
}

// Скачивание 3D модели
function downloadModel() {
    console.log('Функция downloadModel вызвана');
    
    // Получаем имя файла модели из localStorage
    const modelUrl = localStorage.getItem('lastModelUrl');
    console.log('modelUrl из localStorage:', modelUrl);
    
    if (!modelUrl) {
        console.error('URL модели не найден');
        showError('Ошибка: URL модели не найден');
        return;
    }
    
    // Извлекаем имя файла из URL
    const filename = modelUrl.split('/').pop().split('?')[0];
    
    // Загружаем список доступных форматов
    fetch('/api/formats')
        .then(response => response.json())
        .then(data => {
            if (data.formats && data.formats.length > 0) {
                // Показываем модальное окно с выбором формата
                showFormatSelectModal(filename, data.formats);
            } else {
                // Если не удалось получить форматы, используем стандартное скачивание
                const downloadUrl = localStorage.getItem('lastDownloadUrl');
                if (downloadUrl) {
                    console.log('Скачивание модели по URL:', downloadUrl);
                    window.open(downloadUrl, '_blank');
                } else {
                    // Создаем URL для скачивания на основе модели
                    let newDownloadUrl = modelUrl;
                    
                    // Добавляем параметр download=true для указания серверу, что это скачивание
                    newDownloadUrl = newDownloadUrl.includes('?') ? 
                        `${newDownloadUrl}&download=true` : 
                        `${newDownloadUrl}?download=true`;
                    
                    console.log('Скачивание модели по созданному URL:', newDownloadUrl);
                    window.open(newDownloadUrl, '_blank');
                }
            }
        })
                .catch(error => {
            console.error('Ошибка при получении списка форматов:', error);
            console.error('Полная информация об ошибке:', JSON.stringify(error, Object.getOwnPropertyNames(error)));
            
            // В случае ошибки используем стандартное скачивание
            const downloadUrl = localStorage.getItem('lastDownloadUrl');
            console.log('downloadUrl из localStorage:', downloadUrl);
            
            if (downloadUrl) {
                console.log('Скачивание модели по URL:', downloadUrl);
                window.open(downloadUrl, '_blank');
            } else {
                console.error('URL для скачивания не найден в localStorage');
                showError('Ошибка при получении списка форматов');
            }
        });
}

// Показать модальное окно выбора формата
function showFormatSelectModal(filename, formats) {
    console.log('Показываем модальное окно выбора формата для файла:', filename);
    console.log('Доступные форматы:', formats);
    
    const modalOverlay = document.getElementById('format-select-modal-overlay');
    if (!modalOverlay) {
        console.error('Элемент format-select-modal-overlay не найден в DOM');
        return;
    }
    
    const formatsContainer = document.getElementById('formats-container');
    if (!formatsContainer) {
        console.error('Элемент formats-container не найден в DOM');
        return;
    }
    
    // Очищаем контейнер форматов
    formatsContainer.innerHTML = '';
    
    // Добавляем карточки для каждого формата
    formats.forEach(format => {
        const card = document.createElement('div');
        card.className = 'format-card';
        card.dataset.format = format.id;
        
        // Создаем иконку формата
        const icon = document.createElement('div');
        icon.className = 'format-icon';
        icon.textContent = format.extension.replace('.', '').toUpperCase();
        
        // Создаем название формата
        const name = document.createElement('div');
        name.className = 'format-name';
        name.textContent = format.name;
        
        // Создаем расширение формата
        const ext = document.createElement('div');
        ext.className = 'format-ext';
        ext.textContent = format.extension;
        
        // Добавляем элементы в карточку
        card.appendChild(icon);
        card.appendChild(name);
        card.appendChild(ext);
        
        // Добавляем обработчик клика
        card.addEventListener('click', () => {
            // Удаляем выделение со всех карточек
            document.querySelectorAll('.format-card').forEach(c => c.classList.remove('selected'));
            
            // Выделяем выбранную карточку
            card.classList.add('selected');
            
            // Конвертируем и скачиваем файл
            convertAndDownload(filename, format.id);
        });
        
        // Добавляем карточку в контейнер
        formatsContainer.appendChild(card);
    });
    
    // Показываем модальное окно
    modalOverlay.style.display = 'flex';
    modalOverlay.style.justifyContent = 'center';
    modalOverlay.style.alignItems = 'center';
}

// Закрыть модальное окно выбора формата
function closeFormatSelectModal() {
    const modalOverlay = document.getElementById('format-select-modal-overlay');
    if (modalOverlay) {
        modalOverlay.style.display = 'none';
    } else {
        console.error('Элемент format-select-modal-overlay не найден');
    }
}

// Конвертировать и скачать файл
function convertAndDownload(filename, format) {
    // Показываем индикатор загрузки
    const progressContainer = document.getElementById('progress-container');
    const progressText = document.getElementById('progress-text');
    
    if (progressContainer && progressText) {
        progressContainer.style.display = 'block';
        progressText.textContent = `Конвертация в формат ${format.toUpperCase()}...`;
    }
    
    // Закрываем модальное окно
    closeFormatSelectModal();
    
    // Отправляем запрос на конвертацию
    fetch(`/api/convert/${filename}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            format: format,
            prompt: localStorage.getItem('lastPrompt') || 'Без описания'
        })
    })
    .then(response => response.json())
    .then(data => {
        // Скрываем индикатор загрузки
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
        
        if (data.success) {
            // Создаем временную ссылку для скачивания
            const link = document.createElement('a');
            link.href = `/download_converted/${data.filename}`;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else {
            console.error('Ошибка при конвертации:', data.error);
            showError(`Ошибка при конвертации: ${data.error}`);
        }
    })
    .catch(error => {
        // Скрываем индикатор загрузки
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
        
        console.error('Ошибка при конвертации:', error);
        showError('Ошибка при конвертации');
    });
}

// Функция для отображения ошибки
function showError(message) {
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    // НЕ скрываем прогресс-контейнер при ошибке
    // Пусть индикатор продолжает крутиться
    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) {
        progressContainer.style.display = 'block';
    }
}

// Переключение на форму регистрации
function switchToRegister() {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('error-message-modal-reg').style.display = 'none';
    document.getElementById('register-form').style.display = 'block';
    if (globallang == "ru"){
        document.getElementById('modal-title').innerText = 'Регистрация';
        document.getElementById('register-username').placeholder = 'Логин';
        document.getElementById('register-password').placeholder = 'Пароль';
        document.getElementById('register-password-confirm').placeholder = 'Повторите пароль';
        document.getElementById('register-email').placeholder = 'E-mail';
    } else {
        document.getElementById('modal-title').innerText = 'Registration';
        document.getElementById('register-username').placeholder = 'Login';
        document.getElementById('register-password').placeholder = 'Password';
        document.getElementById('register-password-confirm').placeholder = 'Confirm password';
        document.getElementById('register-email').placeholder = 'E-mail';
    }
}

// Закрытие модального окна
function closeModal() {
    document.getElementById('modal-overlay').style.display = 'none';
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('modal-title').innerText = 'Вход';
    
    // Очистка полей формы
    document.getElementById('login-username').value = '';
    document.getElementById('login-password').value = '';
    document.getElementById('register-username').value = '';
    document.getElementById('register-password').value = '';
    document.getElementById('register-password-confirm').value = '';
    document.getElementById('register-email').value = '';
    document.getElementById('remember-me').checked = false;
    document.getElementById('error-message-modal-log').style.display = 'none';
    document.getElementById('error-message-modal-reg').style.display = 'none';
}

// Переключение на форму логина
function switchToLogin() {
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('error-message-modal-log').style.display = 'none';
    document.getElementById('login-form').style.display = 'block';
    if (globallang == "ru"){
        document.getElementById('modal-title').innerText = 'Вход';
        document.getElementById('login-username').placeholder = 'Логин';
        document.getElementById('login-password').placeholder = 'Пароль';
    } else {
        document.getElementById('modal-title').innerText = 'Sign in';
        document.getElementById('login-username').placeholder = 'Login';
        document.getElementById('login-password').placeholder = 'Password';
    }
    
}

// Функция для обновления интерфейса после логина
function updateUserUI(username) {
    const accountLink = document.getElementById('account-link');
    accountLink.onclick = toggleUserMenu;
    
    // Создаем выпадающее меню
    let userMenu = document.getElementById('user-menu');
    if (!userMenu) {
        userMenu = document.createElement('div');
        userMenu.id = 'user-menu';
        userMenu.style.display = 'none';
        userMenu.style.position = 'absolute';
        userMenu.style.top = '40px';
        userMenu.style.right = '20px';
        userMenu.style.background = 'var(--bg-secondary)';
        userMenu.style.boxShadow = '0px 4px 12px rgba(0,0,0,0.2)';
        userMenu.style.borderRadius = '8px';
        userMenu.style.padding = '15px';
        userMenu.style.zIndex = '1000';
        userMenu.style.minWidth = '180px';

        userMenu.innerHTML = `
            <button id="profile-button" class="btn-secondary" style="display:block; width:100%; margin-bottom:10px;">Личный кабинет</button>
            <button id="logout-button" class="btn-secondary" style="display:block; width:100%;">Выйти</button>
        `;
        document.querySelector('.header-right').appendChild(userMenu);

        // Кнопка выхода
        document.getElementById('logout-button').onclick = () => {
            fetch('/logout')
                .then(() => {
                    //alert('Вы вышли из системы');
                    window.location.reload();
                })
                .catch(error => console.error('Ошибка при выходе:', error));
        };

        // Кнопка "Личный кабинет"
        document.getElementById('profile-button').onclick = () => {
            window.location.href = '/account';
        };
    }
}

// Переключение видимости меню пользователя
function toggleUserMenu() {
    console.log('Переключение видимости меню пользователя');
    const userMenu = document.getElementById('user-menu');
    if (userMenu.style.display === 'none') {
        userMenu.style.display = 'block';
    } else {
        userMenu.style.display = 'none';
    }
}

// Обновляем UI после логина
function submitLogin() {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const rememberMe = document.getElementById('remember-me').checked;

    fetch('/auth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            action: 'login',
            username: username,
            password: password,
            remember_me: rememberMe,
        }),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new TypeError("Ожидался JSON!");
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                window.location.reload();
                updateUserUI(username);
                closeModal();

                if (rememberMe) {
                    sessionStorage.setItem('remembered', 'true');
                } else {
                    sessionStorage.setItem('loggedIn', 'true');
                }
            } else {
                const msg = document.getElementById("error-message-modal-log");
                msg.style.display = "block";
                if (globallang == "ru") {
                    msg.textContent = "Неверный логин или пароль"; 
                } else {
                    msg.textContent = "Incorrect login or password";
                }
                //alert(data.message || 'Неверный логин или пароль');
            }
        })
        .catch(error => {
            // console.error('Ошибка авторизации:', error);
            // alert('Произошла ошибка при авторизации. Пожалуйста, попробуйте позже.');
            const msg = document.getElementById("error-message-modal-log");
            msg.style.display = "block";
            if (globallang == "ru") {
                msg.textContent = "Неверный логин или пароль"; 
            } else {
                msg.textContent = "Incorrect login or password";
            }
        });
}

// После регистрации автоматически логиним пользователя
function submitRegister() {
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;
    const passwordConfirm = document.getElementById('register-password-confirm').value;
    const email = document.getElementById('register-email').value;

    if (password !== passwordConfirm) {
        //alert('Пароли не совпадают');
        const msg = document.getElementById("error-message-modal-reg");
        msg.style.display = "block";
        if (globallang == "ru") {
            msg.textContent = "Пароли не совпадают"; 
        } else {
            msg.textContent = "Passwords don't match";
        }
        return;
    }

    console.log('Отправка запроса на регистрацию:', {
        action: 'register',
        username: username,
        email: email
    });

    fetch('/auth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            action: 'register',
            username: username,
            password: password,
            email: email,
        }),
    })
        .then(response => {
            console.log('Получен ответ от сервера:', {
                status: response.status,
                statusText: response.statusText,
                headers: Object.fromEntries(response.headers.entries())
            });
            return response.text().then(text => {
                console.log('Тело ответа:', text);
                // Пытаемся распарсить как JSON только если это действительно JSON
                if (text.trim().startsWith('{')) {
                    return JSON.parse(text);
                }
                throw new Error('Сервер вернул HTML вместо JSON');
            });
        })
        .then(data => {
            console.log('Успешно распарсили JSON:', data);
            if (data.success) {
                // Показываем окно подтверждения кода вместо автоматического входа
                showConfirmationModal(username, email, password);
                
                // Закрываем модальное окно с регистрацией
                closeModal();
            } else {
                //alert(data.message || 'Ошибка регистрации');
                const msg = document.getElementById("error-message-modal-reg");
                msg.style.display = "block";
                if (globallang == "ru") {
                    msg.textContent = "Ошибка при регистрации. Проверьте данные"; 
                } else {
                    msg.textContent = "Error during registration. Check your data";
                }
            }
        })
        .catch(error => console.error('Ошибка при регистрации:', error));
}

// Функция для отображения окна подтверждения кода
function showConfirmationModal(username, email, password) {
    // Если окна еще нет в DOM, добавляем его
    if (!document.getElementById('confirmation-modal')) {
        const confirmationModal = document.createElement('div');
        confirmationModal.className = 'confirmation-modal';
        confirmationModal.id = 'confirmation-modal';
        
        confirmationModal.innerHTML = `
            <div class="confirmation-content">
                <h2 class="confirmation-title">Подтверждение почты</h2>
                <p class="confirmation-message">На вашу почту ${email} отправлен код подтверждения. 
                Пожалуйста, введите его ниже для завершения регистрации.</p>
                
                <div class="code-input-container">
                    <input type="text" maxlength="1" class="code-input" data-index="0">
                    <input type="text" maxlength="1" class="code-input" data-index="1">
                    <input type="text" maxlength="1" class="code-input" data-index="2">
                    <input type="text" maxlength="1" class="code-input" data-index="3">
                    <input type="text" maxlength="1" class="code-input" data-index="4">
                    <input type="text" maxlength="1" class="code-input" data-index="5">
                </div>
                
                <button class="confirm-button">Подтвердить</button>
                
                <div class="resend-code">Отправить код повторно</div>
            </div>
        `;
        
        document.body.appendChild(confirmationModal);
        
        // Добавляем обработчики событий после создания элементов
        document.querySelector('.confirm-button').addEventListener('click', () => {
            submitConfirmationCode(username, password);
        });
        
        document.querySelector('.resend-code').addEventListener('click', () => {
            resendConfirmationCode(email);
        });
        
        // Добавляем обработчики для полей ввода кода
        setupCodeInputHandlers(); // <-- Добавляем эту строку!
    } else {
        // Обновляем информацию в уже существующем окне
        document.querySelector('.confirmation-message').textContent = 
            `На вашу почту ${email} отправлен код подтверждения. Пожалуйста, введите его ниже для завершения регистрации.`;
            
        // Обновляем обработчик кнопки
        document.querySelector('.confirm-button').onclick = () => submitConfirmationCode(username, password);
        
        // Обновляем обработчик для повторной отправки кода
        document.querySelector('.resend-code').onclick = () => resendConfirmationCode(email);
        
        // Также обновляем обработчики для полей ввода
        setupCodeInputHandlers(); // <-- И здесь тоже!
    }
    
    // Показываем окно
    document.getElementById('confirmation-modal').style.display = 'flex';
}

// Настройка обработчиков для полей ввода кода
function setupCodeInputHandlers() {
    const codeInputs = document.querySelectorAll('.code-input');
    
    codeInputs.forEach((input, index) => {
        // При вводе цифры автоматически переходим к следующему полю
        input.addEventListener('input', function(e) {
            if (this.value.length === 1) {
                const nextInput = document.querySelector(`.code-input[data-index="${index + 1}"]`);
                if (nextInput) {
                    nextInput.focus();
                }
            }
        });
        
        // Обработка клавиши Backspace
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Backspace' && this.value.length === 0) {
                const prevInput = document.querySelector(`.code-input[data-index="${index - 1}"]`);
                if (prevInput) {
                    prevInput.focus();
                }
            }
        });
    });
}

// Отправка кода подтверждения на сервер
function submitConfirmationCode(username, password) {
    // Собираем код из всех полей ввода
    const codeInputs = document.querySelectorAll('.code-input');
    let confirmationCode = '';
    
    codeInputs.forEach(input => {
        confirmationCode += input.value;
    });
    
    // Проверяем, что введены все 6 цифр
    if (confirmationCode.length !== 6) {
        alert('Пожалуйста, введите полный 6-значный код');
    
        return;
    }
    
    // Отправляем запрос на подтверждение кода
    fetch('/verify_code', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            username: username,
            code: confirmationCode
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // После успешного подтверждения скрываем окно и делаем вход
            document.getElementById('confirmation-modal').style.display = 'none';
            
            // Автоматически выполняем вход пользователя
            fetch('/auth', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: 'login',
                    username: username,
                    password: password,
                    remember_me: false,
                }),
            })
            .then(response => response.json())
            .then(loginData => {
                if (loginData.success) {
                    updateUserUI(username);
                    sessionStorage.setItem('loggedIn', 'true');
                } else {
                    console.error('Ошибка при автоматическом входе:', loginData.message);
                    alert('Подтверждение успешно, но автоматический вход не удался. Пожалуйста, войдите вручную.');
                    openModal();
                    switchToLogin();
                }
            })
            .catch(error => {
                console.error('Ошибка автоматического входа:', error);
                alert('Подтверждение успешно, но автоматический вход не удался. Пожалуйста, войдите вручную.');
                openModal();
                switchToLogin();
            });
        } else {
            // Сообщаем об ошибке
            alert(data.message || 'Неверный код подтверждения');
        }
    })
    .catch(error => {
        console.error('Ошибка при проверке кода:', error);
        alert('Произошла ошибка при проверке кода. Пожалуйста, попробуйте еще раз.');
    });
}

// Функция для повторной отправки кода подтверждения
function resendConfirmationCode(email) {
    // Отправляем запрос на повторную отправку кода
    console.log('Отправка запроса на повторную отправку кода:', email);
    fetch('/auth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            action: 'resend_code',
            email: email
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Новый код подтверждения отправлен на вашу почту');
            
            // Очищаем поля ввода кода
            document.querySelectorAll('.code-input').forEach(input => {
                input.value = '';
            });
            
            // Фокусируемся на первом поле
            document.querySelector('.code-input[data-index="0"]').focus();
        } else {
            alert(data.message || 'Не удалось отправить новый код');
        }
    })
    .catch(error => {
        console.error('Ошибка при запросе нового кода:', error);
        alert('Произошла ошибка при запросе нового кода. Пожалуйста, попробуйте позже.');
    });
}

// Открытие модального окна
function openModal() {
    document.getElementById('modal-overlay').style.display = 'flex';
    if (globallang == "ru"){
        document.getElementById('modal-title').innerText = 'Вход';
        document.getElementById('login-username').placeholder = 'Логин';
        document.getElementById('login-password').placeholder = 'Пароль';
    } else {
        document.getElementById('modal-title').innerText = 'Sign in';
        document.getElementById('login-username').placeholder = 'Login';
        document.getElementById('login-password').placeholder = 'Password';
    }
}

// Словарь для переводов
const translations = {
    ru: {
        ideaLabel: "Идея:",
        ideaText: "Красная сфера с радиусом 1",
        promptGuide: "Гайд по промптам",
        downloadButton: "Скачать 3D модель",
        profileButton: "Личный кабинет",
        logoutButton: "Выйти",
        searchPlaceholder: "Введите промпт...",
        generateButton: "Генерировать",
        accountLink: "Аккаунт",
        myAccount: "Мой кабинет",
        modalTitle: "Вход",
        loginButton: "Войти",
        registerButton: "Зарегистрироваться",
        backButton: "Назад",
        rememberLabel: "Запомнить меня",
        forgotLink: "Забыли пароль?",
        subsLink: "Подписки"
    },
    en: {
        ideaLabel: "Idea:",
        ideaText: "Red sphere with radius 1",
        promptGuide: "Prompt Guide",
        downloadButton: "Download 3D Model",
        profileButton: "Profile",
        logoutButton: "Logout",
        searchPlaceholder: "Enter prompt...",
        generateButton: "Generate",
        accountLink: "Account",
        myAccount: "My Account",
        modalTitle: "Login",
        loginButton: "Sign In",
        registerButton: "Register",
        backButton: "Back",
        rememberLabel: "Remember me",
        forgotLink: "Forgot password?",
        subsLink: "Subscriptions"
    }
};

// Функция для установки языка
function setLanguage(lang) {
    console.log('[DEBUG] Смена языка на:', lang);
    const t = translations[lang] || translations.ru;
    globallang = lang;
    console.log("Установлен язык в глобальную переменную: " + globallang);
    
    // Обновляем тексты
    document.getElementById('idea-label').textContent = t.ideaLabel;
    document.getElementById('idea-text').textContent = t.ideaText;
    
    // Гайд по промптам
    const promptGuide = document.querySelector('.prompt-guide');
    if (promptGuide) promptGuide.textContent = t.promptGuide;
    
    // Поле ввода и кнопки
    document.getElementById('search-input').placeholder = t.searchPlaceholder;
    
    // Элементы модального окна
    document.getElementById('modal-title').textContent = t.modalTitle;
    document.getElementById('login-button').textContent = t.loginButton;
    document.getElementById('switch-to-register').textContent = t.registerButton;
    document.getElementById('register-button').textContent = t.registerButton;
    document.getElementById('switch-to-login').textContent = t.backButton;
    document.getElementById('remember-label').textContent = t.rememberLabel;
    document.getElementById('forgot-link').textContent = t.forgotLink;
    document.getElementById('subs-link').textContent = t.subsLink;
    
    // Проверяем, существуют ли эти элементы для выпадающего меню
    const profileButton = document.getElementById('profile-button');
    const logoutButton = document.getElementById('logout-button');
    
    if (profileButton) profileButton.textContent = t.profileButton;
    if (logoutButton) logoutButton.textContent = t.logoutButton;
    
    // Обновляем ссылку на аккаунт
    const accountLink = document.getElementById('account-link');
    if (accountLink) {
        // Если пользователь авторизован, показываем "Мой кабинет"/"My Account"
        if (accountLink.getAttribute('data-authenticated') === 'true') {
            accountLink.textContent = t.myAccount;
        } else if (accountLink.textContent === 'Аккаунт' || accountLink.textContent === 'Account') {
            accountLink.textContent = t.accountLink;
        }
    }

    // Отправляем выбор языка на сервер
    fetch('/set_language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang }),
    })
    .then(() => {
        // После смены языка проверяем статус авторизации
        return fetch('/auth_status');
    })
    .then(response => response.json())
    .then(data => {
        const accountLink = document.getElementById('account-link');
        if (data.authenticated) {
            // Если пользователь авторизован, обновляем интерфейс
            accountLink.textContent = t.myAccount;
            updateUserUI(data.username);
        } else {
            // Если пользователь не авторизован, отображаем "Аккаунт"
            accountLink.textContent = t.accountLink;
            accountLink.onclick = openModal;
        }
    })
    .catch(error => console.error('Ошибка при смене языка или проверке авторизации:', error));
}

// Функция для очистки данных о генерации
function clearGenerationData() {
    localStorage.removeItem('lastPrompt');
    localStorage.removeItem('lastModelUrl');
    localStorage.removeItem('lastDownloadUrl');
    localStorage.removeItem('generationStartTime');
    localStorage.removeItem('generationPhase');
    lastGeneratedModelUrl = null;
    lastDownloadUrl = null;
}

// Функция для проверки доступности локального сервера генерации
function checkLocalServerAvailability() {
    const proxyUrl = "/proxy/health";
    console.log('Проверка доступности локального сервера через прокси:', proxyUrl);
    
    fetch(proxyUrl)
    .then(response => {
        if (response.ok) {
            console.log('Локальный сервер доступен!');
            return response.json();
        } else {
            throw new Error(`Сервер вернул статус ${response.status}`);
        }
    })
    .then(data => {
        console.log('Информация о сервере:', data);
        // Сохраняем информацию о доступности GPU
        localStorage.setItem('gpuAvailable', data.gpu_available || false);
    })
    .catch(error => {
        console.error('Ошибка при проверке локального сервера:', error);
        // Показываем сообщение об ошибке
        showError('Локальный сервер генерации недоступен. Убедитесь, что скрипт shape_server.py запущен на вашем ПК.');
    });
}

// Функция для открытия модального окна восстановления пароля
function openRecoveryModal() {
    const modal = document.getElementById('password-recovery-modal');
    modal.style.display = 'flex';
    setTimeout(() => {
        modal.classList.add('active');
    }, 10);
    
    // Закрываем окно авторизации
    document.getElementById('modal-overlay').style.display = 'none';
    if (globallang == "ru") {
        document.getElementById('password-recovery-title').textContent = "Восстановление пароля";
        document.getElementById('password-recovery-message').textContent = "Введите ваш никнейм для восстановления пароля";
        document.getElementById('recovery-username').placeholder = "Никнейм";
        document.getElementById('password-recovery-button').textContent = "Восстановить пароль";
    }   else {
        document.getElementById('password-recovery-title').textContent = "Password recovery";
        document.getElementById('password-recovery-message').textContent = "Enter your nickname to recover your password";
        document.getElementById('recovery-username').placeholder = "Nickname";
        document.getElementById('password-recovery-button').textContent = "Recover password";
    }
}

// Функция для закрытия модального окна восстановления пароля
function closeRecoveryModal() {
    const modal = document.getElementById('password-recovery-modal');
    document.getElementById('correct-message-modal-recovery').style.display = 'none';
    modal.classList.remove('active');
    document.getElementById('error-message-modal-log').style.display = 'none';
    document.getElementById('error-message-modal-reg').style.display = 'none';
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300);
}

// Функция для отправки запроса на восстановление пароля
function submitPasswordRecovery() {
    const username = document.getElementById('recovery-username').value.trim();
    
    if (!username) {
        //alert('Пожалуйста, введите никнейм');
        const msg = document.getElementById("error-message-modal-recovery");
        msg.style.display = "block";
        if (globallang == "ru") {
            msg.textContent = "Введите логин"; 
        } else {
            msg.textContent = "Enter login";
        }
        return;
    }
    console.log('Отправка запроса на восстановление пароля:', username);
    // Отправляем запрос на сервер
    fetch('/reset_password', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || `Ошибка сервера: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            //alert('Новый пароль отправлен на вашу почту');
            const msg = document.getElementById('correct-message-modal-recovery');
            msg.style.display = 'block';
            if (globallang == "ru") {
                msg.textContent = "Новый пароль отправлен на вашу почту " + data.email;
            } else {
                msg.textContent = "New password was sent to your email " + data.email;
            }
            document.getElementById('error-message-modal-recovery').style.display = 'none';
            setTimeout(() => {
                closeRecoveryModal();
                openModal();
            }, 10000);
            // Открываем окно входа
        } else {
            //alert(data.message || 'Ошибка при восстановлении пароля');
            const msg = document.getElementById("error-message-modal-recovery");
            msg.style.display = "block";
            if (globallang == "ru") {
                msg.textContent = "Некорректный логин"; 
            } else {
                msg.textContent = "Incorrect login";
            }
        }
    })
    .catch(error => {
        console.error('Ошибка при восстановлении пароля:', error);
        //alert('Произошла ошибка при восстановлении пароля. Пожалуйста, попробуйте позже.');
        const msg = document.getElementById("error-message-modal-recovery");
        msg.style.display = "block";
        if (globallang == "ru") {
            msg.textContent = "Некорректный логин"; 
        } else {
            msg.textContent = "Incorrect login";
        }
    });
}

// Обновляем обработчик для ссылки "Забыли пароль?"
document.addEventListener('DOMContentLoaded', function() {
    const forgotLink = document.getElementById('forgot-link');
    if (forgotLink) {
        forgotLink.onclick = function(e) {
            e.preventDefault();
            openRecoveryModal();
        };
    }
});

// Проверка статуса авторизации при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Проверяем авторизацию
    checkAuthStatus();
    
    // ... existing code ...
});

// Функция проверки статуса авторизации
function checkAuthStatus() {
    fetch('/auth_status')
        .then(response => response.json())
        .then(data => {
            const accountLink = document.getElementById('account-link');
            if (accountLink) {
                if (data.authenticated) {
                    accountLink.textContent = 'Мой кабинет';
                    setAccountLinkAuthStatus(true);
                } else {
                    accountLink.textContent = 'Аккаунт';
                    accountLink.setAttribute('href', 'javascript:void(0)');
                    accountLink.setAttribute('onclick', 'openModal()');
                    setAccountLinkAuthStatus(false);
                }
            }
        })
        .catch(error => {
            console.error('Ошибка при проверке авторизации:', error);
        });
}

// Функция для установки статуса аутентификации на account-link
function setAccountLinkAuthStatus(isAuthenticated) {
    const accountLink = document.getElementById('account-link');
    if (accountLink) {
        accountLink.setAttribute('data-authenticated', isAuthenticated ? 'true' : 'false');
    }
}

function handleSubscriptionsClick() {
    fetch('/auth_status')
        .then(r => r.json())
        .then(data => {
            if (!data.authenticated) {
                openModal();
            } else {
                window.location.href = '/subscriptions';
            }
        });
}

// Функция для перевода текста с русского на английский
async function translateToEnglish(text) {
    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        if (!response.ok) {
            console.error('Ошибка перевода:', data.error);
            showError(data.error || 'Ошибка перевода');
            return text; // Возвращаем оригинальный текст в случае ошибки
        }
        
        return data.translated_text;
    } catch (error) {
        console.error('Ошибка при переводе:', error);
        showError('Ошибка при отправке запроса на перевод');
        return text; // Возвращаем оригинальный текст в случае ошибки
    }
}

async function getCurrentLanguage(need_update = "True") {

    const response = await fetch('/set_language', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    });
    const data = await response.json();
    const lang = data.lang;
    
    if (need_update == "True") {
        setTimeout(() => {
            setLanguage(lang); 
        }, 500);
    }
}

