const texts = {
  en: {
    headerTitle: "Modelit",
    account: "Account",
    placeholder: "Enter the prompt...",
    ideaLabel: "Idea:",
    ideaText: "A red sphere with a radius of 1",
    promptGuide: "Prompt Guide",
    resultBox: "Result will be here",
    modalTitleLogin: "Login",
    loginPlaceholderUser: "Username",
    loginPlaceholderPass: "Password",
    rememberLabel: "Remember me",
    forgotPass: "Forgot password?",
    loginBtn: "Login",
    registerBtn: "Register",
    modalTitleRegister: "Register",
    regPlaceholderUser: "Username",
    regPlaceholderPass: "Password",
    regPlaceholderPass2: "Confirm Password",
    regPlaceholderEmail: "Email",
    registerDo: "Register",
    registerBack: "Back",
    footer: "© Pyan Inc. 2025",
    genErrorEmpty: "Please enter a description.",
    genErrorProcess: "Generating model...",
    genErrorFail: "Error generating model.",
    genErrorLoad: "Error loading 3D model.",
    genSuccess: "Generated!"
    
  },
  ru: {
    headerTitle: "Modelit",
    account: "Аккаунт",
    placeholder: "Введите промпт...",
    ideaLabel: "Идея:",
    ideaText: "Красная сфера с радиусом 1",
    promptGuide: "Гайд по промптам",
    resultBox: "Здесь будет результат",
    modalTitleLogin: "Вход",
    loginPlaceholderUser: "Логин",
    loginPlaceholderPass: "Пароль",
    rememberLabel: "Запомнить меня",
    forgotPass: "Забыли пароль?",
    loginBtn: "Войти",
    registerBtn: "Зарегистрироваться",
    modalTitleRegister: "Регистрация",
    regPlaceholderUser: "Логин",
    regPlaceholderPass: "Пароль",
    regPlaceholderPass2: "Повторите пароль",
    regPlaceholderEmail: "Email",
    registerDo: "Зарегистрироваться",
    registerBack: "Назад",
    footer: "© Pyan Inc. 2025",
    genErrorEmpty: "Пожалуйста, введите описание модели.",
    genErrorProcess: "Генерация модели...",
    genErrorFail: "Ошибка при генерации модели.",
    genErrorLoad: "Ошибка при загрузке 3D модели.",
    genSuccess: "Сгенерировано!"
  }
};

let currentLang = 'ru';

function setLanguage(lang) {
  currentLang = lang;
  const t = texts[lang];
  document.getElementById('header-title').textContent = t.headerTitle;
  document.getElementById('account-link').textContent = t.account;
  document.getElementById('search-input').placeholder = t.placeholder;
  document.getElementById('idea-label').textContent = t.ideaLabel;
  document.getElementById('idea-text').textContent = t.ideaText;
  document.getElementById('prompt-guide').textContent = t.promptGuide;
  document.getElementById('result-box').textContent = t.resultBox;
  document.querySelector('.footer').textContent = t.footer;

  document.getElementById('modal-title').textContent = t.modalTitleLogin;
  document.getElementById('login-username').placeholder = t.loginPlaceholderUser;
  document.getElementById('login-password').placeholder = t.loginPlaceholderPass;
  document.getElementById('remember-label').textContent = t.rememberLabel;
  document.getElementById('forgot-link').textContent = t.forgotPass;
  document.getElementById('login-button').textContent = t.loginBtn;
  document.getElementById('switch-to-register').textContent = t.registerBtn;

  document.getElementById('register-username').placeholder = t.regPlaceholderUser;
  document.getElementById('register-password').placeholder = t.regPlaceholderPass;
  document.getElementById('register-password-confirm').placeholder = t.regPlaceholderPass2;
  document.getElementById('register-email').placeholder = t.regPlaceholderEmail;
  document.getElementById('register-button').textContent = t.registerDo;
  document.getElementById('switch-to-login').textContent = t.registerBack;
}

function openModal() {
const modalOverlay = document.getElementById('modal-overlay');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const modalTitle = document.getElementById('modal-title');

modalOverlay.style.display = 'flex';
showLogin();

function showLogin() {
  loginForm.style.display = 'block';
  registerForm.style.display = 'none';
  modalTitle.textContent = texts[currentLang].modalTitleLogin;
}

function showRegister() {
  loginForm.style.display = 'none';
  registerForm.style.display = 'block';
  modalTitle.textContent = texts[currentLang].modalTitleRegister;
}
}
window.onload = function() {
  setLanguage('ru');

  const modalOverlay = document.getElementById('modal-overlay');
  const loginForm = document.getElementById('login-form');
  const registerForm = document.getElementById('register-form');
  const modalTitle = document.getElementById('modal-title');
  
  function openModal() {
    modalOverlay.style.display = 'flex';
    showLogin();
  }

  function closeModal() {
    modalOverlay.style.display = 'none';
  }

  modalOverlay.addEventListener('click', function(e) {
    if (e.target === modalOverlay) {
      closeModal();
    }
  });

  document.getElementById('switch-to-register').addEventListener('click', showRegister);
  document.getElementById('switch-to-login').addEventListener('click', showLogin);

  function showLogin() {
    loginForm.style.display = 'block';
    registerForm.style.display = 'none';
    modalTitle.textContent = texts[currentLang].modalTitleLogin;
  }

  function showRegister() {
    loginForm.style.display = 'none';
    registerForm.style.display = 'block';
    modalTitle.textContent = texts[currentLang].modalTitleRegister;
  }

  const searchInput = document.getElementById('search-input');
  const errorMessage = document.getElementById('error-message');
  const resultBox = document.getElementById('result-box');
  const downloadLink = document.getElementById('download-link');
  const downloadUrl = document.getElementById('download-url');
  const searchBar = document.getElementById('search-bar');
  const generateButton = document.getElementById('generate-button');
  const button = document.getElementById('download-button');

  generateButton.addEventListener('click', () => {
    const text = searchInput.value.trim();
    if (text) {
      generateModel();
    } else {
      alert('Введите текст для генерации модели!');
    }
  });

  let modelDownloadUrl = ''; // Переменная для сохранения ссылки на модель

  function generateModel() {
  const t = texts[currentLang];
  const text = searchInput.value.trim();
  
  if (!text) {
      if (errorMessage) errorMessage.innerText = t.genErrorEmpty;
      return;
  }

  if (errorMessage) errorMessage.innerText = t.genErrorProcess; // "Генерация модели..."*/
  //if (resultBox) resultBox.innerHTML = '<div class="loader"></div>';

  fetch('/generate', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: text }),
  })
      .then((r) => r.json())
      .then((data) => {
          if (data.error) {
              if (errorMessage) errorMessage.innerText = data.error;
              if (resultBox) resultBox.textContent = t.genErrorFail;
              return;
          }

          const modelUrl = data.model_url;
          modelDownloadUrl = modelUrl; // Сохраняем ссылку для скачивания

          visualizeModel(modelUrl, () => {
              if (errorMessage) errorMessage.innerText = t.genSuccess; // "Сгенерировано!"
          });
      })
      .catch((err) => {
          console.error(err);
          if (errorMessage) errorMessage.innerText = t.genErrorFail;
      });
}

  function downloadModel() {
      if (modelDownloadUrl) {
          window.location.href = modelDownloadUrl; // Перенаправляем на ссылку для скачивания
      } else {
          alert('Модель ещё не сгенерирована!');
      }
  }

  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault(); 
      generateModel();
    }
  });

  searchBar.addEventListener('click', function(e) {
    if (e.target.id !== 'search-input') {
      generateModel();
    }
  });

  function visualizeModel(url, callback) {
    const t = texts[currentLang];
    
    resultBox.innerHTML = '';
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, resultBox.clientWidth / resultBox.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(resultBox.clientWidth, resultBox.clientHeight);
    resultBox.appendChild(renderer.domElement);

    scene.background = new THREE.Color(0x333333);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    const loader = new THREE.PLYLoader();
    loader.load(
      url,
      function (geometry) {
        if (!geometry || geometry.attributes.position.count === 0) {
          console.error('Empty geometry');
          resultBox.textContent = t.genErrorFail;
          callback(false);
          return;
        }
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          metalness: 0.5,
          roughness: 0.5
        });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
        camera.position.set(0, 0, 5);
        mesh.scale.multiplyScalar(1); // Увеличим для наглядности
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        mesh.position.sub(center);

        camera.position.z = 5;
        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        function animate() {
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        }
        animate();
        const downloadButton = document.createElement('button');
          downloadButton.id = 'download-button';
          downloadButton.style.position = 'absolute';
          downloadButton.style.top = '10px';
          downloadButton.style.right = '10px';
          downloadButton.style.width = '40px';
          downloadButton.style.height = '40px';
          downloadButton.style.background = 'rgba(255, 255, 255, 0.1)';
          downloadButton.style.borderRadius = '50%';
          downloadButton.style.border = 'none';
          downloadButton.style.cursor = 'pointer';
          downloadButton.style.zIndex = '10';
          downloadButton.style.backgroundImage = `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="%23ffffff" d="M256 0a256 256 0 1 0 0 512A256 256 0 1 0 256 0zM244.7 395.3l-112-112c-4.6-4.6-5.9-11.5-3.5-17.4s8.3-9.9 14.8-9.9l64 0 0-96c0-17.7 14.3-32 32-32l32 0c17.7 0 32 14.3 32 32l0 96 64 0c6.5 0 12.3 3.9 14.8 9.9s1.1 12.9-3.5 17.4l-112 112c-6.2 6.2-16.4 6.2-22.6 0z"/></svg>')`;
          downloadButton.style.backgroundRepeat = 'no-repeat';
          downloadButton.style.backgroundPosition = 'center';
          downloadButton.style.backgroundSize = '18px 18px';  

          downloadButton.onclick = () => {
              if (url) {
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = 'model.ply'; // Укажите имя файла
                  link.click();
              } else {
                  alert('Модель не найдена!');
              }
          };
          resultBox.appendChild(downloadButton);
        callback(true);
      },
      function (xhr) {
        console.log( ( xhr.loaded / xhr.total * 100 ) + '% загружено' );
      },
      function (error) {
        console.error('Ошибка при загрузке PLY файла:', error);
        errorMessage.innerText = t.genErrorLoad;
        callback(false);
      }
    );
  }
};