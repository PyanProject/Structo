import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import os
import argparse
import pickle
from skimage import measure
import objaverse
import clip  # Убедитесь, что вы установили библиотеку (pip install git+https://github.com/openai/CLIP.git)

# Dataset for потоковой загрузки Objaverse-XL объектов без полного скачивания.
# Датасет получает список UID через функцию objaverse.load_uids() и загружает по одному объекту через load_objects().
class ObjaverseVoxelDataset(Dataset):
    def __init__(self, split="train", voxel_size=64, limit=None, filter_prompt=None):
        self.voxel_size = voxel_size
        # Загружаем UID всех объектов
        self.uids = objaverse.load_uids()
        # Попытаемся загрузить метаданные, если функция доступна, иначе пропускаем фильтрацию.
        try:
            self.metadata = objaverse.load_metadata()
        except AttributeError:
            print("Warning: objaverse.load_metadata is not available. Skipping filtering.")
            self.metadata = None

        # Если передан фильтр и метаданные доступны, оставляем только те объекты, описание которых содержит искомый текст
        if filter_prompt and self.metadata is not None:
            filter_lower = filter_prompt.lower()
            self.uids = [uid for uid in self.uids if filter_lower in self.metadata.get(uid, {}).get("description", "").lower()]
        
        if limit is not None:
            self.uids = self.uids[:limit]

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        # Load the selected object by UID.
        # The function returns a dictionary {uid: file_path} (URL or local path).
        objs = objaverse.load_objects([uid])
        file_path = objs[uid]
        mesh = trimesh.load(file_path, force='mesh')

        # Ensure we can voxelize the object. First check if voxelized() is callable.
        if not callable(getattr(mesh, 'voxelized', None)):
            # Попытаемся получить выпуклую оболочку, если объект является PointCloud или если voxelized отсутствует.
            try:
                mesh = mesh.convex_hull
            except Exception:
                pass

        # Если после convex_hull метод voxelized всё ещё недоступен, используем альтернативную вокселизацию.
        if not callable(getattr(mesh, 'voxelized', None)):
            from trimesh.voxel import creation as voxel_creation
            pitch = mesh.bounding_box.extents.max() / self.voxel_size
            try:
                voxel_obj = voxel_creation.voxelize(mesh, pitch)
            except Exception as e:
                raise AttributeError("Loaded object doesn't support voxelized() even with alternative method.") from e
        else:
            pitch = mesh.bounding_box.extents.max() / self.voxel_size
            voxel_obj = mesh.voxelized(pitch)

        voxels = voxel_obj.matrix.astype(np.float32)
        grid = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=voxels.dtype)
        v_shape = voxels.shape
        slices_voxels = []
        slices_grid = []
        for d, s in enumerate(v_shape):
            if s <= self.voxel_size:
                grid_start = (self.voxel_size - s) // 2
                grid_end = grid_start + s
                vox_start = 0
                vox_end = s
            else:
                vox_start = (s - self.voxel_size) // 2
                vox_end = vox_start + self.voxel_size
                grid_start = 0
                grid_end = self.voxel_size
            slices_voxels.append(slice(vox_start, vox_end))
            slices_grid.append(slice(grid_start, grid_end))
        grid[slices_grid[0], slices_grid[1], slices_grid[2]] = voxels[slices_voxels[0], slices_voxels[1], slices_voxels[2]]
        voxel_tensor = torch.tensor(grid).unsqueeze(0)
        prompt = self.get_object_prompt(uid)
        return voxel_tensor, prompt

    def get_object_prompt(self, uid):
        # Если метаданные недоступны, возвращаем "default object"
        if self.metadata is None:
            return "default object"
        # Иначе возвращаем описание по uid или значение по умолчанию
        return self.metadata.get(uid, {}).get("description", "default object")

# Voxel encoder using 3D convolutions
class VoxelEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VoxelEncoder, self).__init__()
        # Input: (batch, 1, 64, 64, 64)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)   # -> (32, 32, 32, 32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)   # -> (64, 16, 16, 16)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)  # -> (128, 8, 8, 8)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # -> (256, 4, 4, 4)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(256*4*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4*4, latent_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Conditional voxel decoder – принимает латентный вектор и условный вектор (например, от CLIP)
class ConditionalVoxelDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, voxel_size):
        super(ConditionalVoxelDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim + cond_dim, 256*4*4*4)
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)  # -> (128, 8, 8, 8)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)   # -> (64, 16, 16, 16)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)    # -> (32, 32, 32, 32)
        self.deconv4 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)     # -> (1, 64, 64, 64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.voxel_size = voxel_size
        
    def forward(self, z, cond):
        # Объединяем латентный и текстовый векторы
        z_cond = torch.cat([z, cond], dim=1)
        x = self.fc(z_cond)
        x = x.view(-1, 256, 4, 4, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.sigmoid(self.deconv4(x))
        return x

# Conditional VAE with GAN loss – CVAE_Conditional
class CVAE_Conditional(nn.Module):
    def __init__(self, latent_dim, voxel_size, cond_dim):
        super(CVAE_Conditional, self).__init__()
        self.encoder = VoxelEncoder(latent_dim)
        self.decoder = ConditionalVoxelDecoder(latent_dim, cond_dim, voxel_size)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, cond):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)
        return recon, mu, logvar

# Discriminator network for voxel grids (GAN part)
class Discriminator(nn.Module):
    def __init__(self, voxel_size):
        super(Discriminator, self).__init__()
        # Input: (batch, 1, voxel_size, voxel_size, voxel_size)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(256*4*4*4, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return self.sigmoid(x)

# VAE loss: reconstruction loss + KL divergence
def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# Training procedure for VAE-GAN on Objaverse-XL
def train(args, device):
    # Используем датасет для Objaverse-XL с потоковой загрузкой
    dataset = ObjaverseVoxelDataset(split="train", voxel_size=args.voxel_size, limit=args.limit, filter_prompt=args.prompt)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your objaverse configuration.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    latent_dim = 128
    cond_dim = 512  # размерность текстового эмбеддинга CLIP (для модели ViT-B/32 обычно 512)
    cvae = CVAE_Conditional(latent_dim, args.voxel_size, cond_dim).to(device)
    discriminator = Discriminator(args.voxel_size).to(device)
    
    optimizer_G = optim.Adam(cvae.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    cvae.train()
    discriminator.train()
    
    # Загружаем CLIP-модель один раз для условного кодирования
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model_clip.eval()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for voxels, prompts in dataloader:
            voxels = voxels.to(device)
            # Получаем условный эмбеддинг для батча промптов через CLIP
            tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                cond = model_clip.encode_text(tokens)  # (batch_size, cond_dim)
            
            # -------------------------
            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(voxels.size(0), 1, device=device)
            fake_labels = torch.zeros(voxels.size(0), 1, device=device)
            
            recon, mu, logvar = cvae(voxels, cond)
            D_real = discriminator(voxels)
            loss_D_real = nn.functional.binary_cross_entropy(D_real, real_labels)
            D_fake = discriminator(recon.detach())
            loss_D_fake = nn.functional.binary_cross_entropy(D_fake, fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # -------------------------
            # Train Generator (VAE part with adversarial loss)
            optimizer_G.zero_grad()
            recon, mu, logvar = cvae(voxels, cond)
            loss_vae = vae_loss(recon, voxels, mu, logvar)
            D_fake_forG = discriminator(recon)
            loss_adv = nn.functional.binary_cross_entropy(D_fake_forG, real_labels)
            loss_G = loss_vae + args.lambda_adv * loss_adv
            loss_G.backward()
            optimizer_G.step()
            
            epoch_loss += loss_G.item()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(dataset):.4f}")
    
    # Save trained model checkpoint
    checkpoint = {
        "cvae_state_dict": cvae.state_dict(),
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"Training completed. Model checkpoint saved to {args.checkpoint}")

# Generation procedure: generate a new 3D model from random latent vector
def generate(args, device):
    latent_dim = 128
    cvae = CVAE_Conditional(latent_dim, args.voxel_size, args.cond_dim).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cvae.load_state_dict(checkpoint["cvae_state_dict"])
    cvae.eval()
    
    # Sample random latent vector
    z = torch.randn(1, latent_dim).to(device)
    if args.prompt is not None:
        # Кодируем промпт с помощью CLIP (без дообучения, веса CLIP заморожены)
        cond = encode_prompt(args.prompt, device)
    else:
        # Если промпт не задан, используем нулевой вектор
        cond = torch.zeros(1, 512).to(device)
    with torch.no_grad():
        voxel_out = cvae.decoder(z, cond)  # (1, 1, voxel_size, voxel_size, voxel_size)
    voxel_grid = voxel_out.squeeze().cpu().numpy()
    
    # Determine appropriate threshold level for marching cubes
    v_min, v_max = voxel_grid.min(), voxel_grid.max()
    if not (v_min <= 0.5 <= v_max):
         level = (v_min + v_max) / 2
         print(f"Warning: level 0.5 not in voxel grid range [{v_min:.3f}, {v_max:.3f}]. Using level {level:.3f} instead.")
    else:
         level = 0.5
         
    verts, faces, normals, values = measure.marching_cubes(voxel_grid, level=level)
    with open(args.output, "w") as f:
        for v in verts:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))
    print(f"3D model generated and saved to {args.output}")
    # Visualize generated mesh in a window
    mesh_vis = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh_vis.visual.face_colors = [200, 200, 200, 255]
    scene = mesh_vis.scene()
    scene.show()

def prompt_to_latent(prompt, latent_dim):
    # Простой способ: используем хеш промпта для генерации псевдослучайного вектора
    np.random.seed(abs(hash(prompt)) % (2**32))
    latent = np.random.randn(1, latent_dim)
    return torch.tensor(latent, dtype=torch.float32)

def encode_prompt(prompt, device):
    # Используем CLIP для получения текстового эмбеддинга
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        prompt_embedding = model_clip.encode_text(text)
    # Преобразуем эмбеддинг в нужный размер (например, проведем линейное преобразование)
    # Это преобразование нужно будет обучать совместно с VAE-GAN
    return prompt_embedding

def main():
    parser = argparse.ArgumentParser(description="VAE-GAN for 3D Model Generation on Objaverse-XL")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], required=True, help="Mode: train or generate")
    # Аргумент --dataset проигнорирован, используется библиотека objaverse
    parser.add_argument("--dataset", type=str, default="Objaverse-XL", help="(Ignored) Using objaverse API for data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="vae_gan.pth", help="Checkpoint file path")
    parser.add_argument("--voxel_size", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--output", type=str, default="model.obj", help="Output OBJ file for generation")
    parser.add_argument("--lambda_adv", type=float, default=0.001, help="Weight factor for adversarial loss")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for conditional 3D generation")
    parser.add_argument("--cond_dim", type=int, default=512, help="Dimension of the condition vector")
    # Параметр для ограничения количества загружаемых объектов (опционально)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of objects to load")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "train":
        train(args, device)
    elif args.mode == "generate":
        generate(args, device)

if __name__ == "__main__":
    main()
