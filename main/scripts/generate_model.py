#generate_model.py
import torch
import os
import open3d as o3d
from utils.clip_embedding import EmbeddingGenerator
from utils.gan_models import Generator
from utils.mesh_generator import generate_3d_scene_from_embedding

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
    
    generator = Generator(noise_dim=100, embedding_dim=512).to(device)
    checkpoint_path = "models/checkpoints/generator.pth"
    if os.path.exists(checkpoint_path):
        generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
        generator.eval()
        print("[GENERATE] GAN модель успешно загружена.")
    else:
        print("[GENERATE] Модель GAN не найдена.")
        return
    
    while True:
        text = input("[GENERATE] Введите текст для генерации 3D-модели (или 'exit' для выхода): ")
        if text.lower() in ["exit", "quit"]:
            print("[GENERATE] Завершение работы...")
            break
        try:
            embedding = embedding_generator.generate_embedding(text).to(device)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
                
            with torch.no_grad():
                noise = 0.1 * torch.randn(1, generator.noise_dim).to(device)
                generated_data = generator(noise, embedding).cpu().detach().numpy().squeeze()
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(generated_data)
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
            print("[GENERATE] Визуализация облака точек...")
            o3d.visualization.draw_geometries([pcd], window_name="Generated Points")
            
            scene_filename = generate_3d_scene_from_embedding(generated_data, text)
            print(f"[GENERATE] Модель сохранена: {scene_filename}")
        except Exception as e:
            print(f"[GENERATE] Ошибка генерации: {e}")

if __name__ == "__main__":
    main()
