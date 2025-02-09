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
        # Load mesh from .off file
        mesh = trimesh.load(file_path, file_type='off', force='mesh')
        
        # Determine pitch for voxelization
        pitch = mesh.bounding_box.extents.max() / self.voxel_size
        try:
            voxel_obj = mesh.voxelized(pitch)
        except Exception as e:
            try:
                mesh = mesh.convex_hull
                voxel_obj = mesh.voxelized(pitch)
            except Exception as e:
                raise ValueError("Cannot voxelize mesh: " + file_path) from e
        
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
        object_name = os.path.splitext(os.path.basename(file_path))[0]  # extract file name without extension
        object_name = object_name.replace('_', ' ')  # replace underscores with spaces
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
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Encode text prompt using CLIP model
def encode_prompt(prompt, device):
    model_clip, _ = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        prompt_embedding = model_clip.encode_text(tokens)
    return prompt_embedding

# Training procedure for CVAE-GAN on custom dataset
def train(args, device):
    dataset = TestVoxelDataset(root_dir=args.dataset, voxel_size=args.voxel_size)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your dataset directory.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    latent_dim = args.latent_dim
    cond_dim = args.cond_dim
    cvae = CVAE_Conditional(latent_dim, args.voxel_size, cond_dim).to(device)
    discriminator = Discriminator(args.voxel_size).to(device)
    
    optimizer_G = optim.Adam(cvae.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    cvae.train()
    discriminator.train()
    
    # Load CLIP model for text encoding
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model_clip.eval()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for voxels, prompts in dataloader:
            voxels = voxels.to(device)
            tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                cond = model_clip.encode_text(tokens)  # (batch_size, cond_dim)
            
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
            
            # Train Generator (VAE part with adversarial loss)
            optimizer_G.zero_grad()
            recon, mu, logvar = cvae(voxels, cond)
            loss_vae_val = vae_loss(recon, voxels, mu, logvar)
            D_fake_forG = discriminator(recon)
            loss_adv = nn.functional.binary_cross_entropy(D_fake_forG, real_labels)
            loss_G_total = loss_vae_val + args.lambda_adv * loss_adv
            loss_G_total.backward()
            optimizer_G.step()
            
            epoch_loss += loss_G_total.item()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(dataset):.4f}")
    
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

def main():
    parser = argparse.ArgumentParser(description="VAE-GAN for text-to-3D Model Generation on custom dataset")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], required=True, help="Mode: train or generate")
    parser.add_argument("--dataset", type=str, default="testdataset", help="Path to dataset directory containing OFF files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="vae_gan_test.pth", help="Checkpoint file path")
    parser.add_argument("--voxel_size", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--output", type=str, default="model.obj", help="Output OBJ file for generation")
    parser.add_argument("--lambda_adv", type=float, default=0.001, help="Weight factor for adversarial loss")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for conditional 3D generation")
    parser.add_argument("--cond_dim", type=int, default=512, help="Dimension of the condition vector")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of latent vector")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "train":
        train(args, device)
    elif args.mode == "generate":
        generate(args, device)

if __name__ == "__main__":
    main()
