import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh
from PIL import Image
import os
import numpy as np
import re
from datetime import datetime

def print_gpu_memory(message=""):
    """Вывод информации о текущем использовании памяти GPU."""
    if torch.cuda.is_available():
        print(f"\n=== GPU Memory Usage {message} ===")
        print(f"Выделено: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Кэшировано: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print("=" * 30)

def check_gpu():
    """Проверка доступности и информации о GPU."""
    if torch.cuda.is_available():
        print("=== GPU Information ===")
        print(f"CUDA доступен: {torch.cuda.is_available()}")
        print(f"Текущее CUDA устройство: {torch.cuda.current_device()}")
        print(f"Название устройства: {torch.cuda.get_device_name(0)}")
        print(f"Количество доступных GPU: {torch.cuda.device_count()}")
        print(f"Память GPU:")
        print(f" - Всего: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f" - Выделено: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f" - Кэшировано: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print("===================")
        return True
    else:
        print("CUDA недоступен. Будет использован CPU.")
        return False

def sanitize_filename(prompt):
    """Преобразует промпт в безопасное имя файла."""
    # Заменяем пробелы на подчеркивания и удаляем специальные символы
    safe_name = re.sub(r'[^\w\s-]', '', prompt.lower())
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    # Добавляем временную метку для уникальности
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_{timestamp}"

def main():
    # Проверка GPU
    has_gpu = check_gpu()
    
    # Создаем директорию для результатов
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Определяем устройство
    device = torch.device('cuda' if has_gpu else 'cpu')
    print(f"Using device: {device}")

    # Оптимизации для CUDA
    if has_gpu:
        torch.backends.cuda.matmul.allow_tf32 = True  # Ускорение матричных операций
        torch.backends.cudnn.benchmark = True  # Оптимизация производительности
        # Очистка кэша CUDA
        torch.cuda.empty_cache()
        print_gpu_memory("после очистки кэша")

    # Загружаем модели
    print("\nLoading models...")
    xm = load_model('transmitter', device=device)
    print_gpu_memory("после загрузки transmitter")
    
    model = load_model('text300M', device=device)
    print_gpu_memory("после загрузки text300M")
    
    diffusion = diffusion_from_config(load_config('diffusion'))
    print_gpu_memory("после загрузки diffusion")

    # Параметры генерации
    batch_size = 1
    guidance_scale = 15.0
    
    # Запрашиваем промпт у пользователя
    print("\nВведите текстовый промпт для генерации 3D модели")
    print("Например: 'a red chair', 'a blue car', 'a wooden table'")
    prompt = input("Ваш промпт: ").strip()
    
    if not prompt:
        print("Промпт не может быть пустым. Используем промпт по умолчанию: 'a red chair'")
        prompt = "a red chair"
    
    print(f"\nНачинаем генерацию для промпта: '{prompt}'")

    try:
        # Генерация латентного представления
        print("\nSampling latents...")
        with torch.cuda.amp.autocast(enabled=has_gpu):  # Автоматическое смешанное вычисление
            latents = sample_latents(
                batch_size=batch_size,
                model=model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[prompt] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=has_gpu,  # Используем fp16 только при наличии GPU
                use_karras=True,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            print_gpu_memory("после генерации латентов")

        # Сохранение результатов
        for i, latent in enumerate(latents):
            # Создаем уникальное имя файла на основе промпта
            safe_name = sanitize_filename(prompt)
            
            # Сохранение меша
            print(f"\nСохранение меша {i+1}/{batch_size}...")
            mesh = decode_latent_mesh(xm, latent).tri_mesh()
            print_gpu_memory("после декодирования меша")
            
            mesh_filename = f"{safe_name}.obj"
            mesh_path = os.path.join(output_dir, mesh_filename)
            with open(mesh_path, 'w') as f:
                mesh.write_obj(f)
            print(f"Меш сохранен как: {mesh_filename}")

            # Создание и сохранение GIF
            print(f"\nСоздание превью {i+1}/{batch_size}...")
            cameras = create_pan_cameras(64, device)
            images = decode_latent_images(xm, latent, cameras, rendering_mode='nerf')
            print_gpu_memory("после рендеринга превью")
            
            # Преобразование изображений в формат PIL
            frames = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    if img.shape[-1] == 3:  # если каналы в конце
                        img = img
                    elif img.shape[0] == 3:  # если каналы в начале
                        img = np.transpose(img, (1, 2, 0))
                    img = Image.fromarray(img)
                frames.append(img)
            
            # Сохранение GIF
            gif_filename = f"{safe_name}.gif"
            gif_path = os.path.join(output_dir, gif_filename)
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=100,
                loop=0
            )
            print(f"Превью сохранено как: {gif_filename}")

            # Очистка памяти GPU после каждой итерации
            if has_gpu:
                torch.cuda.empty_cache()
                print_gpu_memory("после очистки кэша")

        print(f"\nГенерация завершена! Результаты сохранены в папке '{output_dir}':")
        print(f"- 3D модель: {mesh_filename}")
        print(f"- Превью: {gif_filename}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU память переполнена. Попробуйте уменьшить размер batch_size или karras_steps")
            if has_gpu:
                torch.cuda.empty_cache()
                print_gpu_memory("после ошибки")
        raise e

if __name__ == "__main__":
    main() 