// Функции для страницы личного кабинета
globallang = "";
// Словарь переводов для личного кабинета
const accountTranslations = {
    ru: {
        accountTitle: "Личный кабинет",
        loading: "Загрузка...",
        myModels: "Мои модели",
        settings: "Настройки",
        noModels: "У вас пока нет сгенерированных моделей",
        createFirst: "Создайте свою первую модель в генераторе",
        createModel: "Создать модель",
        view: "Просмотр",
        download: "Скачать",
        unknownDate: "Неизвестная дата",
        changePassword: "Изменить пароль",
        currentPassword: "Текущий пароль",
        newPassword: "Новый пароль",
        confirmPassword: "Подтвердите пароль",
        passwordChanged: "Пароль успешно изменен",
        passwordError: "Ошибка при изменении пароля",
        fillAllFields: "Пожалуйста, заполните все поля",
        passwordsDontMatch: "Новые пароли не совпадают",
        tryLater: "Произошла ошибка, попробуйте позже",
        saveSettings: "Сохранить настройки",
        settingsSaved: "Настройки успешно сохранены",
        settingsError: "Ошибка при сохранении настроек",
        displayName: "Отображаемое имя",
        yourName: "Ваше имя",
        accountSettings: "Настройки аккаунта",
        dangerZone: "Опасная зона",
        warningText: "Эти действия нельзя отменить. Будьте осторожны!",
        logout: "Выйти из аккаунта",
        modelsTab: "Мои модели",
        settingsTab: "Настройки",
        genLink: "Генератор",
        supText: "Если у вас возникли вопросы или проблемы, наша команда поддержки всегда готова помочь!",
        supLink: "Поддержка",
        supMail: "Связаться с поддержкой",
        ForgotPassword: "Забыли пароль?"
    },
    en: {
        accountTitle: "My Account",
        loading: "Loading...",
        myModels: "My Models",
        settings: "Settings",
        noModels: "You have no generated models yet",
        createFirst: "Create your first model in the generator",
        createModel: "Create Model",
        view: "View",
        download: "Download",
        unknownDate: "Unknown date",
        changePassword: "Change Password",
        currentPassword: "Current Password",
        newPassword: "New Password",
        confirmPassword: "Confirm Password",
        passwordChanged: "Password changed successfully",
        passwordError: "Error changing password",
        fillAllFields: "Please fill in all fields",
        passwordsDontMatch: "New passwords do not match",
        tryLater: "An error occurred, please try later",
        saveSettings: "Save Settings",
        settingsSaved: "Settings saved successfully",
        settingsError: "Error saving settings",
        displayName: "Display Name",
        yourName: "Your name",
        accountSettings: "Account Settings",
        dangerZone: "Danger Zone",
        warningText: "These actions cannot be undone. Be careful!",
        logout: "Logout",
        modelsTab: "My Models",
        settingsTab: "Settings",
        genLink: "Generator",
        supText: "If you have any questions or problems, our support team is always ready to help!",
        supLink: "Support",
        supMail: "Contact support",
        ForgotPassword: "Forgot password?"
    }
};

// Функция для установки языка на странице личного кабинета
function setAccountLanguage(lang) {
    const t = accountTranslations[lang] || accountTranslations.ru;
    // Заголовок
    const title = document.querySelector('.account-title');
    if (title) title.textContent = t.accountTitle;
    // Имя пользователя и email (если еще не загружены)
    const username = document.getElementById('profile-username');
    if (username && (!username.textContent || username.textContent === 'Загрузка...' || username.textContent === 'Loading...')) username.textContent = t.loading;
    const email = document.getElementById('profile-email');
    if (email && (!email.textContent || email.textContent === 'Загрузка...' || email.textContent === 'Loading...')) email.textContent = t.loading;
    // Вкладки
    const tabBtns = document.querySelectorAll('.tab-btn');
    if (tabBtns.length > 1) {
        tabBtns[0].textContent = t.modelsTab;
        tabBtns[1].textContent = t.settingsTab;
    }
    // Пустой блок моделей
    const modelsEmpty = document.getElementById('models-empty');
    if (modelsEmpty) {
        const h3 = modelsEmpty.querySelector('h3');
        if (h3) h3.textContent = t.noModels;
        const p = modelsEmpty.querySelector('p');
        if (p) p.textContent = t.createFirst;
        const btn = modelsEmpty.querySelector('a.btn-primary');
        if (btn) btn.textContent = t.createModel;
    }
    // Настройки
    const settingsTab = document.getElementById('settings-tab');
    if (settingsTab) {
        const sectionTitles = settingsTab.querySelectorAll('.section-title');
        if (sectionTitles.length > 0) sectionTitles[0].textContent = t.changePassword;
        if (sectionTitles.length > 1) sectionTitles[1].textContent = t.accountSettings;
        if (sectionTitles.length > 2) sectionTitles[2].textContent = t.dangerZone;
        // Лейблы и плейсхолдеры
        const labels = settingsTab.querySelectorAll('label');
        if (labels.length > 0) labels[0].textContent = t.currentPassword;
        if (labels.length > 1) labels[1].textContent = t.newPassword;
        if (labels.length > 2) labels[2].textContent = t.confirmPassword;
        if (labels.length > 3) labels[3].textContent = t.displayName;
        const displayNameInput = document.getElementById('display-name');
        if (displayNameInput) displayNameInput.placeholder = t.yourName;
        // Кнопки
        const changePassBtn = document.getElementById('change-password-btn');
        if (changePassBtn) changePassBtn.textContent = t.changePassword;
        const saveSettingsBtn = document.getElementById('save-settings-btn');
        if (saveSettingsBtn) saveSettingsBtn.textContent = t.saveSettings;
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) logoutBtn.textContent = t.logout;
        // Опасная зона
        const warningText = settingsTab.querySelector('.warning-text');
        if (warningText) warningText.textContent = t.warningText;

        const genBtn = document.getElementById('gen-link');
        genBtn.textContent = t.genLink;
        const supText = document.getElementById('support-text');
        supText.textContent = t.supText;
        const supLink = document.getElementById('support-link');
        supLink.textContent = t.supLink;
        const supMail = document.getElementById('support-mail');
        supMail.textContent = t.supMail;
        const frgtPass = document.getElementById('forgot-password-btn');
        frgtPass.textContent = t.ForgotPassword;
        
    }
    saveCurrentLanguage(lang);
}

// Для этого определяем глобальную переменную accountLang
let accountLang = 'ru';

// Функция для смены языка (например, по кнопке)
function changeAccountLanguage(lang) {
    accountLang = lang;
    setAccountLanguage(lang);
}

// Вызвать setAccountLanguage при загрузке
setAccountLanguage(accountLang);

getCurrentLanguage();

// Ждем загрузку DOM
document.addEventListener('DOMContentLoaded', function() {
    // Скрываем лоадер после загрузки страницы
    const loaderOverlay = document.getElementById('loader-overlay');
    if (loaderOverlay) {
        setTimeout(() => {
            loaderOverlay.style.display = 'none';
        }, 500);
    }

    // Проверка авторизации
    checkAuthStatus();

    // Инициализация вкладок
    initTabs();

    // Загружаем данные пользователя
    loadUserProfile();

    // Загружаем модели пользователя
    loadUserModels();

    // Добавляем обработчики событий для форм
    addFormEventListeners();

    // Подвязываем смену языка к кнопкам RU/EN в личном кабинете
    const ruBtn = document.querySelector('.lang-btn[onclick*="ru"]');
    const enBtn = document.querySelector('.lang-btn[onclick*="en"]');
    if (ruBtn) {
        ruBtn.addEventListener('click', function(e) {
            e.preventDefault();
            changeAccountLanguage('ru');
        });
    }
    if (enBtn) {
        enBtn.addEventListener('click', function(e) {
            e.preventDefault();
            changeAccountLanguage('en');
        });
    }
});

// Проверка статуса авторизации
function checkAuthStatus() {
    fetch('/auth_status')
        .then(response => response.json())
        .then(data => {
            if (!data.authenticated) {
                // Если пользователь не авторизован, перенаправляем на главную страницу
                window.location.href = '/';
            }
        })
        .catch(error => {
            console.error('Ошибка при проверке авторизации:', error);
        });
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
    
    // Отправляем запрос на сервер
    fetch('/reset_password', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username })
    })
    .then(response => response.json())
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
            }, 10000);

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
    const forgotLink = document.getElementById('forgot-password-btn');
    if (forgotLink) {
        forgotLink.onclick = function(e) {
            e.preventDefault();
            openRecoveryModal();
        };
    }
});

function closeRecoveryModal() {
    const modal = document.getElementById('password-recovery-modal');
    document.getElementById('correct-message-modal-recovery').style.display = 'none';
    modal.classList.remove('active');
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300);
}

// Функция для открытия модального окна восстановления пароля
function openRecoveryModal() {
    const modal = document.getElementById('password-recovery-modal');
    modal.style.display = 'flex';
    setTimeout(() => {
        modal.classList.add('active');
    }, 10);

    if (globallang == "ru") {
        document.getElementById('rec-title').textContent = "Восстановление пароля";
        document.getElementById('rec-msg').textContent = "Введите ваш никнейм для восстановления пароля";
        document.getElementById('recovery-username').placeholder = "Никнейм";
        document.getElementById('rec-btn').textContent = "Восстановить пароль";
    }   else {
        document.getElementById('rec-title').textContent = "Password recovery";
        document.getElementById('rec-msg').textContent = "Enter your nickname to recover your password";
        document.getElementById('recovery-username').placeholder = "Nickname";
        document.getElementById('rec-btn').textContent = "Recover password";
    }
    
    // Закрываем окно авторизации
    //document.getElementById('modal-overlay').style.display = 'none';
}

// Инициализация вкладок
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Убираем активные классы у всех кнопок и панелей
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));

            // Добавляем активный класс к нажатой кнопке
            this.classList.add('active');

            // Отображаем соответствующую панель
            const tabId = this.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
}

// Загрузка профиля пользователя
function loadUserProfile() {
    fetch('/user_profile')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('profile-username').textContent = data.username;
                document.getElementById('profile-email').textContent = data.email;
                
                // Если есть отображаемое имя, заполняем его в форму настроек
                if (data.display_name) {
                    document.getElementById('display-name').value = data.display_name;
                }
            } else {
                console.error('Ошибка при загрузке профиля:', data.message);
            }
        })
        .catch(error => {
            console.error('Ошибка при загрузке профиля:', error);
        });
}

// Загрузка моделей пользователя
function loadUserModels() {
    fetch('/user_models')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (data.models && data.models.length > 0) {
                    // Если у пользователя есть модели, скрываем блок "пусто" и отображаем сетку
                    document.getElementById('models-empty').style.display = 'none';
                    renderModelGrid(data.models);
                } else {
                    // Если моделей нет, показываем блок "пусто"
                    document.getElementById('models-empty').style.display = 'block';
                    document.getElementById('models-grid').style.display = 'none';
                }
            } else {
                console.error('Ошибка при загрузке моделей:', data.message);
            }
        })
        .catch(error => {
            console.error('Ошибка при загрузке моделей:', error);
        });
}

// Отрисовка сетки моделей
function renderModelGrid(models) {
    const modelsGrid = document.getElementById('models-grid');
    modelsGrid.style.display = 'grid';
    modelsGrid.innerHTML = ''; // Очищаем сетку

    models.forEach(model => {
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';

        const modelHtml = `
            <div class="model-thumbnail">
                <!-- Превью модели может быть добавлено через Three.js или как изображение -->
            </div>
            <div class="model-info">
                <div class="model-name">${model.prompt || 'Без названия'}</div>
                <div class="model-date">${formatDate(model.created_at)}</div>
                <div class="model-actions">
                    <button class="model-btn view-btn" onclick="viewModel('${model.filename}')">Просмотр</button>
                    <button class="model-btn download-model-btn" onclick="downloadModel('${model.filename}')">Скачать</button>
                </div>
            </div>
        `;

        modelCard.innerHTML = modelHtml;
        modelsGrid.appendChild(modelCard);
    });
}

// Форматирование даты
function formatDate(dateString) {
    if (!dateString) return 'Неизвестная дата';
    
    const date = new Date(dateString);
    return date.toLocaleDateString('ru-RU', {
        day: '2-digit', 
        month: '2-digit', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Просмотр модели
function viewModel(filename) {
    window.location.href = `/view_model?filename=${filename}`;
}

// Скачивание модели
function downloadModel(filename) {
    window.location.href = `/download_model/${filename}`;
}

// Добавление обработчиков событий для форм
function addFormEventListeners() {
    // Изменение пароля
    const changePasswordForm = document.getElementById('change-password-btn');
    if (changePasswordForm) {
        changePasswordForm.addEventListener('click', function() {
            const currentPassword = document.getElementById('current-password').value;
            const newPassword = document.getElementById('new-password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            const messageElement = document.getElementById('password-change-message');
            const t = accountTranslations[accountLang] || accountTranslations.ru;

            // Валидация
            if (!currentPassword || !newPassword || !confirmPassword) {
                messageElement.textContent = t.fillAllFields;
                messageElement.className = 'form-message error';
                return;
            }

            if (newPassword !== confirmPassword) {
                messageElement.textContent = t.passwordsDontMatch;
                messageElement.className = 'form-message error';
                return;
            }

            // Отправка запроса на изменение пароля
            fetch('/change_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    messageElement.textContent = t.passwordChanged;
                    messageElement.className = 'form-message success';
                    // Очищаем поля
                    document.getElementById('current-password').value = '';
                    document.getElementById('new-password').value = '';
                    document.getElementById('confirm-password').value = '';
                } else {
                    messageElement.textContent = data.message || t.passwordError;
                    messageElement.className = 'form-message error';
                }
            })
            .catch(error => {
                console.error('Ошибка при изменении пароля:', error);
                messageElement.textContent = t.tryLater;
                messageElement.className = 'form-message error';
            });
        });
    }

    // Сохранение настроек аккаунта
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', function() {
            const displayName = document.getElementById('display-name').value;
            const messageElement = document.getElementById('settings-message');
            const t = accountTranslations[accountLang] || accountTranslations.ru;

            fetch('/update_profile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    display_name: displayName
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    messageElement.textContent = t.settingsSaved;
                    messageElement.className = 'form-message success';
                    window.location.reload();
                } else {
                    messageElement.textContent = data.message || t.settingsError;
                    messageElement.className = 'form-message error';
                }
            })
            .catch(error => {
                console.error('Ошибка при сохранении настроек:', error);
                messageElement.textContent = t.tryLater;
                messageElement.className = 'form-message error';
            });
        });
    }

    // Выход из аккаунта
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            fetch('/logout')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.href = '/';
                    }
                })
                .catch(error => {
                    console.error('Ошибка при выходе из аккаунта:', error);
                });
        });
    }
} 

async function getCurrentLanguage() {
    const response = await fetch('/set_language', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    });

    const data = await response.json();
    setAccountLanguage(data.lang);
    console.log("Текущий язык: " + data.lang);
}

async function saveCurrentLanguage(lang) {
    const response = await fetch('/set_language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang: lang })
    });
    console.log("Изменены языковые настройки на "+lang);
    const data = await response.json();
    console.log("Язык на сервере: ", data.lang);
    globallang = lang;
}