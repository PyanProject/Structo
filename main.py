import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import os
import argparse
from skimage import measure
import clip
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Dataset for loading .off files from the custom dataset
class TestVoxelDataset(Dataset):
    def __init__(self, root_dir, voxel_size=64, transform=None):
        self.root_dir = root_dir
        self.voxel_size = voxel_size
        self.transform = transform
        self.files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(".off"):
                    self.files.append(os.path.join(dirpath, filename))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Try loading the mesh with process=False to avoid inhomogeneous array issues.
        try:
            mesh = trimesh.load(file_path, file_type='off', force='mesh', process=False)
        except Exception as e:
            try:
                mesh = trimesh.load(file_path, file_type='off', process=False)
            except Exception as e:
                # If loading fails, log a warning and return None, None so that this sample is skipped.
                print(f"Warning: Skipping file: {file_path} due to error: {e}")
                return None, None
        
        # Determine pitch for voxelization
        pitch = mesh.bounding_box.extents.max() / self.voxel_size
        try:
            voxel_obj = mesh.voxelized(pitch)
        except Exception as e:
            try:
                mesh = mesh.convex_hull
                voxel_obj = mesh.voxelized(pitch)
            except Exception as e:
                print(f"Warning: Skipping file: {file_path} due to error during voxelization: {e}")
                return None, None
        
        voxels = voxel_obj.matrix.astype(np.float32)
        
        # Center the voxel grid in a fixed shape tensor
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
        voxel_tensor = torch.tensor(grid).unsqueeze(0)  # (1, voxel_size, voxel_size, voxel_size)
        
        # Generate a prompt based on the file name by extracting the object's name
        object_name = os.path.splitext(os.path.basename(file_path))[0]
        object_name = object_name.replace('_', ' ')
        prompt = "3d model of " + object_name
        
        return voxel_tensor, prompt

# Voxel Encoder using 3D convolutions
class VoxelEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VoxelEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 32, 32, 32)
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

# Conditional Voxel Decoder using 3D transpose convolutions
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
        # Concatenate latent vector with conditional embedding
        z_cond = torch.cat([z, cond], dim=1)
        x = self.fc(z_cond)
        x = x.view(-1, 256, 4, 4, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.sigmoid(self.deconv4(x))
        return x

# CVAE-Conditional model combining the encoder and the decoder
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

# Discriminator for 3D voxel grid
class Discriminator(nn.Module):
    def __init__(self, voxel_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        # Calculate flattened dimension (assumes voxel_size is divisible by 16)
        final_dim = voxel_size // 16
        self.fc = nn.Linear(256 * final_dim * final_dim * final_dim, 1)
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

# VAE loss combining reconstruction loss and KL divergence
def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Encode text prompt using CLIP model
def encode_prompt(prompt, device):
    model_clip, _ = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        prompt_embedding = model_clip.encode_text(tokens)
    return prompt_embedding

# Compute IoU between binary occupancy voxel grids
def compute_iou(recon, target, threshold=0.5):
    """
    Compute IoU between binary occupancy voxel grids.
    recon, target shapes: (batch_size, 1, voxel_size, voxel_size, voxel_size)
    """
    binary_recon = (recon > threshold).float()
    binary_target = (target > threshold).float()
    intersection = (binary_recon * binary_target).view(binary_recon.size(0), -1).sum(dim=1)
    union = ((binary_recon + binary_target) > 0).float().view(binary_recon.size(0), -1).sum(dim=1)
    iou = intersection / (union + 1e-6)
    return iou.mean()

# Training procedure for CVAE-GAN on custom dataset
def train(args, device):
    dataset = TestVoxelDataset(root_dir=args.dataset, voxel_size=args.voxel_size)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your dataset directory.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn_skip_none)
    
    latent_dim = args.latent_dim
    cond_dim = args.cond_dim
    cvae = CVAE_Conditional(latent_dim, args.voxel_size, cond_dim).to(device)
    discriminator = Discriminator(args.voxel_size).to(device)
    
    optimizer_G = optim.Adam(cvae.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr * 0.1)
    
    cvae.train()
    discriminator.train()
    
    # Load CLIP model for text encoding
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model_clip.eval()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        acc_real_total = 0.0
        acc_fake_total = 0.0
        acc_fake_forG_total = 0.0
        acc_iou_total = 0.0
        batch_count = 0

        for voxels, prompts in dataloader:
            voxels = voxels.to(device)
            tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                cond = model_clip.encode_text(tokens)  # (batch_size, cond_dim)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(voxels.size(0), 1, device=device) * 0.9  # label smoothing for real samples
            fake_labels = torch.zeros(voxels.size(0), 1, device=device)
            
            recon, mu, logvar = cvae(voxels, cond)
            D_real = discriminator(voxels)
            loss_D_real = nn.functional.binary_cross_entropy(D_real, real_labels)
            D_fake = discriminator(recon.detach())
            loss_D_fake = nn.functional.binary_cross_entropy(D_fake, fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # Calculate discriminator accuracy for real and fake samples
            acc_real = (D_real >= 0.5).float().mean().item()  # accuracy for real samples
            acc_fake = (D_fake < 0.5).float().mean().item()     # accuracy for fake samples
            
            # Train Generator: perform 2 updates per batch to help generator learning
            total_gen_loss = 0.0
            for _ in range(2):
                optimizer_G.zero_grad()
                recon, mu, logvar = cvae(voxels, cond)
                loss_vae_val = vae_loss(recon, voxels, mu, logvar)
                D_fake_forG = discriminator(recon)
                loss_adv = nn.functional.binary_cross_entropy(D_fake_forG, real_labels)
                # Warm-up: escalate adversarial loss weight gradually over first 5 epochs.
                warmup_factor = min((epoch+1) / 5.0, 1.0)
                # Calculate ratio: if adv loss is much higher than recon loss, increase its influence.
                loss_ratio = loss_adv.item() / (loss_vae_val.item() + 1e-8)
                # Clamp ratio between 0.1 and 10.0 to avoid extreme scaling.
                loss_ratio = max(0.1, min(loss_ratio, 10.0))
                effective_lambda_adv = args.lambda_adv * warmup_factor * loss_ratio
                loss_G_total = args.recon_weight * loss_vae_val + effective_lambda_adv * loss_adv
                loss_G_total.backward()
                optimizer_G.step()
                total_gen_loss += loss_G_total.item()
            # Считаем среднее значение генераторного лосса по двум обновлениям
            loss_G_total = total_gen_loss / 2.0

            # Calculate generator accuracy: percentage of generated (fake) samples fooling the discriminator
            acc_fake_forG = (D_fake_forG >= 0.5).float().mean().item()
            
            # Compute reconstruction IoU as quality metric between generated recon and original voxels
            iou = compute_iou(recon, voxels)
            
            # Accumulate metrics per batch
            acc_real_total += acc_real
            acc_fake_total += acc_fake
            acc_fake_forG_total += acc_fake_forG
            acc_iou_total += iou.item()
            batch_count += 1
            
            epoch_loss += loss_G_total
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(dataset):.4f}, "
              f"Discriminator Acc (Real): {acc_real_total/batch_count:.4f}, "
              f"Discriminator Acc (Fake): {acc_fake_total/batch_count:.4f}, "
              f"Generator Acc: {acc_fake_forG_total/batch_count:.4f}, "
              f"IoU: {acc_iou_total/batch_count:.4f}")
    
    checkpoint = {"cvae_state_dict": cvae.state_dict()}
    torch.save(checkpoint, args.checkpoint)
    print(f"Training completed. Model checkpoint saved to {args.checkpoint}")

# Generation procedure: generate a new 3D model from a text prompt
def generate(args, device, need_visualisation=True):
    latent_dim = args.latent_dim
    cond_dim = args.cond_dim
    cvae = CVAE_Conditional(latent_dim, args.voxel_size, cond_dim).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cvae.load_state_dict(checkpoint["cvae_state_dict"])
    cvae.eval()
    
    # Sample a random latent vector
    z = torch.randn(1, latent_dim).to(device)
    if args.prompt is not None:
        cond = encode_prompt(args.prompt, device)
    else:
        cond = torch.zeros(1, cond_dim).to(device)
    with torch.no_grad():
        voxel_out = cvae.decoder(z, cond)  # (1, 1, voxel_size, voxel_size, voxel_size)
    voxel_grid = voxel_out.squeeze().cpu().numpy()
    # Pre-smooth voxel grid for better connectivity using a Gaussian filter
    voxel_grid = gaussian_filter(voxel_grid, sigma=1)
    
    # Determine threshold level for marching cubes
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
    if need_visualisation:
        # Visualize generated mesh
        mesh_vis = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh_vis.visual.face_colors = [200, 200, 200, 255]
        scene = mesh_vis.scene()
        scene.show()

    return args.output

def collate_fn_skip_none(batch):
    # Filter out samples where voxel_tensor is None.
    filtered = [item for item in batch if item[0] is not None]
    if len(filtered) == 0:
        return None  # или можно выбросить исключение, если все данные некорректны
    return torch.utils.data.dataloader.default_collate(filtered)

def analyze_clip_embeddings(args, device):
    # Load CLIP model and tokenize texts
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Define a list of text prompts to analyze.
    texts = [
        "3d model of chair",
        "3d model of airplane",
        "3d model of table",
        "3d model of car",
        "3d model of sofa"
    ]
    
    # Tokenize texts and compute embeddings.
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
    
    # Convert embeddings to numpy and normalize them.
    text_embeddings = text_embeddings.cpu().numpy()
    normed_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Calculate and print cosine similarity matrix.
    cosine_sim = np.dot(normed_embeddings, normed_embeddings.T)
    print("Cosine similarity matrix:")
    print(cosine_sim)
    
    # Reduce dimensions using PCA for visualization.
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(text_embeddings)
    
    # Plot the 2D PCA projection.
    plt.figure(figsize=(8, 6))
    for i, text in enumerate(texts):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=text)
        plt.text(reduced_embeddings[i, 0] + 0.01, reduced_embeddings[i, 1] + 0.01, text)
    
    plt.title("PCA Projection of CLIP Text Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="VAE-GAN for text-to-3D Model Generation on custom dataset")
    parser.add_argument("--mode", type=str, choices=["train", "generate", "analyze"], required=True, 
                        help="Mode: train, generate or analyze")
    parser.add_argument("--dataset", type=str, default="testdataset", help="Path to dataset directory containing OFF files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for generator")
    parser.add_argument("--checkpoint", type=str, default="vae_gan_test.pth", help="Checkpoint file path")
    parser.add_argument("--voxel_size", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--output", type=str, default="model.obj", help="Output OBJ file for generation")
    parser.add_argument("--lambda_adv", type=float, default=0.001, help="Weight factor for adversarial loss")
    parser.add_argument("--recon_weight", type=float, default=10.0, help="Weight factor for reconstruction loss")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for conditional 3D generation")
    parser.add_argument("--cond_dim", type=int, default=512, help="Dimension of the condition vector")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of latent vector")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    
    args = parser.parse_args()
    # Override checkpoint parameter to always use "vae_gan_test.pth"
    args.checkpoint = "vae_gan_test.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "train":
        train(args, device)
    elif args.mode == "generate":
        generate(args, device)
    elif args.mode == "analyze":
        analyze_clip_embeddings(args, device)

if __name__ == "__main__":
    main()
