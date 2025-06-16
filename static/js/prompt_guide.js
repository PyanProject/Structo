// Custom JavaScript for the prompt guide page

// Translations for the prompt guide page
const promptGuideTranslationsW = {
    ru: {
        pageTitle: "Structo - Гайд по промптам",
        guideTitle: "Как писать промпты для генерации 3D-моделей",
        guideDescription: "Этот гайд поможет вам создавать эффективные запросы для генерации 3D-моделей в Structo. Следуйте этим рекомендациям для получения наилучших результатов.",
        sectionTitle: "Основные параметры",
        colorParam: "<strong>Цвет:</strong> Укажите цвет объекта",
        shapeParam: "<strong>Форма:</strong> Опишите форму объекта.",
        sizeParam: "<strong>Размер:</strong> Укажите размер объекта.",
        examplesTitle: "Примеры промптов",
        example1: "Красный самолет длиной 10м",
        example2: "Двухэтажный дом",
        example3: "Человек из пластилина",
        backButton: "Вернуться на главную",
        copyright: "&copy; Structo, 2024-2025",
        version: "alpha v.1.5"
    },
    en: {
        pageTitle: "Structo - Prompt Guide",
        guideTitle: "How to Write Prompts for 3D Model Generation",
        guideDescription: "This guide will help you create effective prompts for generating 3D models in Structo. Follow these recommendations for the best results.",
        sectionTitle: "Basic Parameters",
        colorParam: "<strong>Color:</strong> Specify the object's color.",
        shapeParam: "<strong>Shape:</strong> Describe the object's shape.",
        sizeParam: "<strong>Size:</strong> Specify the object's size.",
        examplesTitle: "Prompt Examples",
        example1: "Red airplane length 10m",
        example2: "Two-story house",
        example3: "Man from plasticine",
        backButton: "Back to Main Page",
        copyright: "&copy; Structo, 2024-2025",
        version: "alpha v.1.5"
    }
};

getCurrentLanguage();
// Function to set the language
function setLanguage(lang) {
    console.log('[DEBUG] Changing language to:', lang);
    const t = promptGuideTranslationsW[lang];
    
    // Update page title
    document.title = t.pageTitle;
    
    // Update main content
    document.getElementById('guide-title').textContent = t.guideTitle;
    document.getElementById('guide-description').textContent = t.guideDescription;
    
    // Update section titles
    const sectionTitles = document.querySelectorAll('.section-title');
    if (sectionTitles.length >= 1) sectionTitles[0].textContent = t.sectionTitle;
    if (sectionTitles.length >= 2) sectionTitles[1].textContent = t.examplesTitle;
    
    // Update feature list
    const featureItems = document.querySelectorAll('.feature-text');
    if (featureItems.length >= 1) featureItems[0].innerHTML = t.colorParam;
    if (featureItems.length >= 2) featureItems[1].innerHTML = t.shapeParam;
    if (featureItems.length >= 3) featureItems[2].innerHTML = t.sizeParam;
    
    // Update examples
    const exampleTexts = document.querySelectorAll('.example-text');
    if (exampleTexts.length >= 1) exampleTexts[0].textContent = t.example1;
    if (exampleTexts.length >= 2) exampleTexts[1].textContent = t.example2;
    if (exampleTexts.length >= 3) exampleTexts[2].textContent = t.example3;
    
    // Update back button
    const backButton = document.querySelector('.back-button');
    if (backButton) {
        // Keep the SVG icon and update only the text
        const buttonText = backButton.innerHTML.split('</svg>')[1].trim();
        backButton.innerHTML = backButton.innerHTML.replace(buttonText, t.backButton);
    }
    
    // Update footer
    const copyright = document.querySelector('.copyright');
    if (copyright) copyright.innerHTML = t.copyright;
    
    const version = document.querySelector('.version');
    if (version) version.textContent = t.version;
    
    // Update active language button
    const langButtons = document.querySelectorAll('.lang-btn');
    langButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.toLowerCase() === lang.toLowerCase()) {
            btn.classList.add('active');
        }
    });

    saveCurrentLanguage(lang);
}

document.addEventListener('DOMContentLoaded', function() {
    // Устанавливаем язык по умолчанию на 'ru'
    // setLanguage('ru'); // Удалите или закомментируйте эту строку, чтобы убрать начальный запуск

    // Handle language selection
    const langButtons = document.querySelectorAll('.lang-btn');
    langButtons.forEach(button => {
        button.addEventListener('click', function() {
            const lang = this.textContent.trim().toLowerCase();
            setLanguage(lang);
        });
    });
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
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
    setLanguage(data.lang);
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