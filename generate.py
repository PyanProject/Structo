import torch
import os
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from PIL import Image

# Создаем папку для сохранения результатов
os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading models...")
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

# Уменьшаем количество генерируемых образцов и параметры для ускорения
batch_size = 1  # было 4
guidance_scale = 15.0
prompt = "airplane"
print(f"Generating with prompt: '{prompt}'")

print("Sampling latents...")
latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=32,  # было 64
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf'  # you can change this to 'stf'
size = 32  # Уменьшаем размер для ускорения рендеринга (было 64)

print(f"Rendering results using {render_mode} mode...")
cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    print(f"Rendering sample {i+1}/{batch_size}...")
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    
    # Сохраняем каждый кадр
    for j, img in enumerate(images):
        img_path = f"results/sample_{i+1}_frame_{j+1}.png"
        img.save(img_path)
    
    # Создаем и сохраняем GIF
    frames = [Image.fromarray(img) for img in images]
    gif_path = f"results/sample_{i+1}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )
    print(f"Saved GIF to {gif_path}")

print("Generation completed! Results saved in 'results' folder.")