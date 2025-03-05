/**
 * ModelIT - Основной JavaScript файл
 */

document.addEventListener('DOMContentLoaded', function() {
    // Инициализация общих компонентов
    initUI();
    
    // Проверка поддержки WebGL
    checkWebGL();
});

/**
 * Инициализация пользовательского интерфейса
 */
function initUI() {
    // Добавляем обработчики для всех кнопок с классом .btn
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            this.classList.add('btn-active');
            setTimeout(() => {
                this.classList.remove('btn-active');
            }, 200);
        });
    });
    
    // Инициализация всплывающих подсказок
    const toolTips = document.querySelectorAll('[data-tooltip]');
    toolTips.forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltipText = this.getAttribute('data-tooltip');
            
            const tooltip = document.createElement('div');
            tooltip.classList.add('tooltip');
            tooltip.textContent = tooltipText;
            
            document.body.appendChild(tooltip);
            
            const rect = this.getBoundingClientRect();
            tooltip.style.top = rect.bottom + 10 + 'px';
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            
            this.addEventListener('mouseleave', function onMouseLeave() {
                document.body.removeChild(tooltip);
                this.removeEventListener('mouseleave', onMouseLeave);
            });
        });
    });
}

/**
 * Проверка поддержки WebGL
 */
function checkWebGL() {
    if (!window.WebGLRenderingContext) {
        showBrowserWarning("Ваш браузер не поддерживает WebGL, необходимый для отображения 3D моделей.");
        return false;
    }
    
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!gl) {
        showBrowserWarning("WebGL отключен или не поддерживается вашим браузером/видеокартой.");
        return false;
    }
    
    return true;
}

/**
 * Отображение предупреждения о браузере
 */
function showBrowserWarning(message) {
    if (document.querySelector('.browser-warning')) return;
    
    const warning = document.createElement('div');
    warning.classList.add('browser-warning');
    warning.innerHTML = `
        <div class="browser-warning-content">
            <h3>Внимание!</h3>
            <p>${message}</p>
            <p>Попробуйте использовать последние версии Chrome, Firefox или Edge.</p>
            <button class="btn btn-secondary" id="dismiss-warning">Понятно</button>
        </div>
    `;
    
    document.body.appendChild(warning);
    
    document.getElementById('dismiss-warning').addEventListener('click', function() {
        document.body.removeChild(warning);
    });
}

/**
 * Форматирование даты и времени
 */
function formatDateTime(date) {
    const options = { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    
    return new Intl.DateTimeFormat('ru-RU', options).format(date);
}

/**
 * Анимация загрузки контента
 */
function animateContent() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
            }
        });
    }, { threshold: 0.1 });
    
    elements.forEach(element => {
        observer.observe(element);
    });
} 