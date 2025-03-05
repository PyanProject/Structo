/**
 * ModelViewer - Скрипт для просмотра 3D моделей в приложении ModelIT
 */

class ModelViewer {
    constructor(containerId, options = {}) {
        // Элемент-контейнер
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Контейнер с ID ${containerId} не найден`);
            return;
        }

        // Опции по умолчанию
        this.options = Object.assign({
            backgroundColor: 0xf5f5f5,
            defaultModelColor: 0x3498db,
            ambientLightColor: 0xffffff,
            ambientLightIntensity: 0.5,
            directionalLightColor: 0xffffff,
            directionalLightIntensity: 0.8,
            backLightColor: 0xffffff,
            backLightIntensity: 0.3,
            showAxes: true,
            showGrid: true,
            autoRotate: false,
            rotateSpeed: 0.01
        }, options);

        // Флаги состояния
        this.isInitialized = false;
        this.isModelLoaded = false;
        this.isAutoRotating = this.options.autoRotate;

        // Three.js объекты
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.model = null;

        // Инициализируем Three.js
        this.initialize();
    }

    /**
     * Инициализация Three.js
     */
    initialize() {
        if (!window.THREE) {
            console.error('Three.js не загружен');
            return;
        }

        try {
            // Создаем сцену
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(this.options.backgroundColor);

            // Добавляем освещение
            this.addLighting();

            // Настраиваем камеру
            const containerWidth = this.container.clientWidth;
            const containerHeight = this.container.clientHeight || 400;
            this.camera = new THREE.PerspectiveCamera(45, containerWidth / containerHeight, 0.1, 1000);
            this.camera.position.set(2, 2, 5);

            // Создаем рендерер
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setSize(containerWidth, containerHeight);
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.container.appendChild(this.renderer.domElement);

            // Добавляем контроллеры OrbitControls
            if (window.THREE.OrbitControls) {
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.25;
                this.controls.enableZoom = true;
            } else {
                console.warn('OrbitControls не загружен, управление камерой недоступно');
            }

            // Добавляем вспомогательные элементы
            if (this.options.showGrid) {
                const gridHelper = new THREE.GridHelper(10, 10);
                this.scene.add(gridHelper);
            }

            if (this.options.showAxes) {
                const axesHelper = new THREE.AxesHelper(5);
                this.scene.add(axesHelper);
            }

            // Запускаем анимацию
            this.animate();

            // Добавляем обработчик изменения размера окна
            window.addEventListener('resize', this.onWindowResize.bind(this));

            this.isInitialized = true;
            console.log('ModelViewer инициализирован успешно');
        } catch (error) {
            console.error('Ошибка инициализации ModelViewer:', error);
        }
    }

    /**
     * Добавление освещения на сцену
     */
    addLighting() {
        // Ambient light (рассеянный свет)
        const ambientLight = new THREE.AmbientLight(
            this.options.ambientLightColor, 
            this.options.ambientLightIntensity
        );
        this.scene.add(ambientLight);

        // Directional light (направленный свет)
        const directionalLight = new THREE.DirectionalLight(
            this.options.directionalLightColor,
            this.options.directionalLightIntensity
        );
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Back light (задний свет)
        const backLight = new THREE.DirectionalLight(
            this.options.backLightColor,
            this.options.backLightIntensity
        );
        backLight.position.set(-1, -1, -1);
        this.scene.add(backLight);
    }

    /**
     * Анимация сцены
     */
    animate() {
        requestAnimationFrame(this.animate.bind(this));

        // Вращение модели при включенном автовращении
        if (this.isAutoRotating && this.model) {
            this.model.rotation.y += this.options.rotateSpeed;
        }

        // Обновление контроллеров
        if (this.controls) {
            this.controls.update();
        }

        // Рендеринг сцены
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Обработчик изменения размера окна
     */
    onWindowResize() {
        const containerWidth = this.container.clientWidth;
        const containerHeight = this.container.clientHeight || 400;

        this.camera.aspect = containerWidth / containerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(containerWidth, containerHeight);
    }

    /**
     * Загрузка модели из OBJ/PLY/GLTF файла
     * @param {string} modelPath Путь к файлу модели
     * @param {Object} options Дополнительные опции для загрузки
     */
    loadModel(modelPath, options = {}) {
        if (!this.isInitialized) {
            console.error('Нельзя загрузить модель, ModelViewer не инициализирован');
            return;
        }

        // Очищаем предыдущую модель
        if (this.model) {
            this.scene.remove(this.model);
            this.model = null;
            this.isModelLoaded = false;
        }

        // Определяем расширение файла
        const fileExtension = modelPath.split('.').pop().toLowerCase();
        let loader;

        // Выбираем соответствующий загрузчик
        switch (fileExtension) {
            case 'obj':
                if (!window.THREE.OBJLoader) {
                    console.error('OBJLoader не загружен');
                    return;
                }
                loader = new THREE.OBJLoader();
                break;
            case 'ply':
                if (!window.THREE.PLYLoader) {
                    console.error('PLYLoader не загружен');
                    return;
                }
                loader = new THREE.PLYLoader();
                break;
            case 'glb':
            case 'gltf':
                if (!window.THREE.GLTFLoader) {
                    console.error('GLTFLoader не загружен');
                    return;
                }
                loader = new THREE.GLTFLoader();
                break;
            default:
                console.error(`Неподдерживаемый формат файла: ${fileExtension}`);
                return;
        }

        // Показываем индикатор загрузки, если он есть
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }

        // Загружаем модель
        loader.load(
            modelPath,
            (object) => {
                // Обработка успешной загрузки
                try {
                    // В зависимости от типа загрузчика обрабатываем результат по-разному
                    if (loader instanceof THREE.PLYLoader) {
                        // Для PLY файлов
                        const material = new THREE.MeshStandardMaterial({ 
                            color: options.color || this.options.defaultModelColor, 
                            flatShading: true
                        });
                        this.model = new THREE.Mesh(object, material);
                    } else if (loader instanceof THREE.GLTFLoader) {
                        // Для GLTF/GLB файлов
                        this.model = object.scene;
                    } else {
                        // Для OBJ и других форматов
                        this.model = object;
                    }

                    // Центрируем и масштабируем модель
                    this.centerAndScaleModel();

                    // Добавляем модель на сцену
                    this.scene.add(this.model);
                    this.isModelLoaded = true;

                    // Скрываем индикатор загрузки
                    if (loadingIndicator) {
                        loadingIndicator.style.display = 'none';
                    }

                    console.log('Модель успешно загружена:', modelPath);
                } catch (error) {
                    console.error('Ошибка при обработке загруженной модели:', error);
                    if (loadingIndicator) {
                        loadingIndicator.style.display = 'none';
                    }
                }
            },
            // Прогресс загрузки
            (xhr) => {
                const percent = (xhr.loaded / xhr.total) * 100;
                console.log(`Загрузка модели: ${Math.round(percent)}%`);
            },
            // Ошибка загрузки
            (error) => {
                console.error('Ошибка загрузки модели:', error);
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
            }
        );
    }

    /**
     * Центрирование и масштабирование модели
     */
    centerAndScaleModel() {
        if (!this.model) return;

        // Создаем временный объект для расчета bounding box
        const tempObject = this.model.clone();
        const box = new THREE.Box3().setFromObject(tempObject);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        // Центрируем модель
        this.model.position.x = -center.x;
        this.model.position.y = -center.y;
        this.model.position.z = -center.z;

        // Масштабируем модель
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0) {
            const scale = 3 / maxDim;
            this.model.scale.set(scale, scale, scale);
        }
    }

    /**
     * Включение/выключение автоматического вращения
     * @param {boolean} enable Включить/выключить
     */
    toggleAutoRotate(enable) {
        this.isAutoRotating = enable !== undefined ? enable : !this.isAutoRotating;
        return this.isAutoRotating;
    }

    /**
     * Сброс камеры к начальному положению
     */
    resetCamera() {
        if (this.controls) {
            this.controls.reset();
        }
    }

    /**
     * Установка цвета модели (для моделей без текстур)
     * @param {number} color Цвет в формате 0xRRGGBB
     */
    setModelColor(color) {
        if (!this.model) return;

        this.model.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                if (child.material) {
                    child.material.color.set(color);
                }
            }
        });
    }

    /**
     * Сделать скриншот текущего вида
     * @returns {string} Data URL изображения
     */
    takeScreenshot() {
        this.renderer.render(this.scene, this.camera);
        return this.renderer.domElement.toDataURL('image/png');
    }
} 