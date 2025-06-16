// Словарь переводов для страницы подписок
const subsTranslations = {
    ru: {
        title: "Выберите подходящий тариф",
        free: "Бесплатный",
        pro: "Проф",
        business: "Бизнес",
        priceFree: "0₽",
        pricePro: "499₽/мес",
        priceBusiness: "1999₽/мес",
        featuresFree: [
            "50 генераций обычного качества",
            "Облачное хранилище на 5 моделей"
        ],
        featuresPro: [
            "100 генераций лучшего качества",
            "Облачное хранилище на 50 моделей"
        ],
        featuresBusiness: [
            "Неограниченное количество генераций лучшего качества",
            "Приоритетный доступ к вычислительным ресурсам",
            "Корпоративное облачное хранилище на 500 моделей"
        ],
        btnFree: "Ваш тариф",
        btnPro: "Выбрать",
        btnBusiness: "Выбрать",
        back: "Назад к генератору"
    },
    en: {
        title: "Choose your plan",
        free: "Free",
        pro: "Prof",
        business: "Business",
        priceFree: "0₽",
        pricePro: "499₽/mo",
        priceBusiness: "1999₽/mo",
        featuresFree: [
            "50 regular quality generations",
            "Cloud storage for 5 models"
        ],
        featuresPro: [
            "100 high quality generations",
            "Cloud storage for 50 models"
        ],
        featuresBusiness: [
            "Unlimited high quality generations",
            "Priority access to computing resources",
            "Corporate cloud storage for 500 models"
        ],
        btnFree: "Your plan",
        btnPro: "Choose",
        btnBusiness: "Choose",
        back: "Back to generator"
    }
};

let subsLang = 'ru';

getCurrentLanguage();

function setSubsLanguage(lang) {
    subsLang = lang;
    const t = subsTranslations[lang] || subsTranslations.ru;
    document.querySelector('.subs-title').textContent = t.title;
    // Бесплатный тариф
    document.querySelector('.subs-free .subs-name').textContent = t.free;
    document.querySelector('.subs-free .subs-price').textContent = t.priceFree;
    const freeFeatures = document.querySelectorAll('.subs-free .subs-features li');
    freeFeatures.forEach((li, i) => { li.textContent = t.featuresFree[i] || ''; });
    document.querySelector('.subs-free .subs-btn').textContent = t.btnFree;
    // Профи тариф
    document.querySelector('.subs-pro .subs-name').textContent = t.pro;
    document.querySelector('.subs-pro .subs-price').textContent = t.pricePro;
    const proFeatures = document.querySelectorAll('.subs-pro .subs-features li');
    proFeatures.forEach((li, i) => { li.textContent = t.featuresPro[i] || ''; });
    document.querySelector('.subs-pro .subs-btn').textContent = t.btnPro;
    // Бизнес тариф
    document.querySelector('.subs-business .subs-name').textContent = t.business;
    document.querySelector('.subs-business .subs-price').textContent = t.priceBusiness;
    const businessFeatures = document.querySelectorAll('.subs-business .subs-features li');
    businessFeatures.forEach((li, i) => { li.textContent = t.featuresBusiness[i] || ''; });
    document.querySelector('.subs-business .subs-btn').textContent = t.btnBusiness;
    // Кнопка назад
    document.querySelector('.back-btn').textContent = t.back;

    saveCurrentLanguage(lang);
}

document.addEventListener('DOMContentLoaded', function() {
    setSubsLanguage(subsLang);

    fetch('/api/subscription')
        .then(r => r.json())
        .then(data => {
            const subscription = data.subscription || 1;
            console.log(subscription);

            // Массив кнопок и их соответствие тарифу
            const btns = [
                document.querySelector('.subs-free .subs-btn'),
                document.querySelector('.subs-pro .subs-btn'),
                document.querySelector('.subs-business .subs-btn')
            ];
            btns.forEach((btn, idx) => {
                if (!btn) return;
                if (subscription === idx + 1) {
                    btn.textContent = 'Ваш тариф';
                    btn.classList.add('subs-btn-current');
                    btn.disabled = true;
                    btn.style.opacity = '0.5';
                    btn.style.filter = 'grayscale(0.5)';
                    btn.style.cursor = 'default';
                } else {
                    btn.textContent = 'Выбрать';
                    btn.classList.remove('subs-btn-current');
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    btn.style.filter = 'none';
                    btn.style.cursor = 'pointer';
                }
            });

            // Обработчики для кнопок тарифов только если они не disabled
            if (btns[1] && !btns[1].disabled) {
                btns[1].addEventListener('click', function() {
                    window.location.href = '/pay?plan=pro';
                });
            }
            if (btns[2] && !btns[2].disabled) {
                btns[2].addEventListener('click', function() {
                    window.location.href = '/pay?plan=business';
                });
            }
        });
}); 

async function getCurrentLanguage() {
    const response = await fetch('/set_language', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    });

    const data = await response.json();
    setSubsLanguage(data.lang);
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
}