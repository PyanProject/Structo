"""
Conditional VAE with Attention for Text-to-3D Generation.

Requirements:
- A dataset of 3D models (ModelNet40Dataset from dataset.py) with OFF files.
- EmbeddingGenerator (from embedding_generator.py) that produces a text embedding of dimension 512.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
from tqdm import tqdm

from dataset import ModelNet40Dataset, collate_fn
from embedding_generator import EmbeddingGenerator  # Should return text embedding (dim=512)

# Function to compute Chamfer Distance between two point clouds (B, N, 3)
def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer distance between two batches of point clouds.
    pc1, pc2: tensors of shape (B, N, 3)
    """
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)  # (B, N, N, 3)
    dist = torch.norm(diff, dim=-1)  # (B, N, N)
    min1, _ = torch.min(dist, dim=2)  # (B, N)
    min2, _ = torch.min(dist, dim=1)  # (B, N)
    loss = torch.mean(min1) + torch.mean(min2)
    return loss

# KL divergence function for VAE
def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# Encoder: takes a point cloud (B, 4096, 3) and produces mu and logvar (B, latent_dim)
class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Produces (B, 512, 1)
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
    
    def forward(self, x):
        # x: (B, 4096, 3) -> transpose -> (B, 3, 4096)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.global_pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)        # (B, 512)
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# A simple residual MLP block (can be useful for other architectures)
class ResidualMLP(nn.Module):
    def __init__(self, dim):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = self.relu(out)
        return out

# AttentionDecoder using TransformerDecoder
class AttentionDecoder(nn.Module):
    def __init__(self, latent_dim=256, text_dim=512, num_points=1024, num_tokens=16, model_dim=256, num_layers=3):
        super(AttentionDecoder, self).__init__()
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.num_points = num_points
        # Project latent vector to a set of query tokens
        self.latent_to_tokens = nn.Linear(latent_dim, num_tokens * model_dim)
        # Project text embedding to a set of key/value tokens
        self.text_to_tokens = nn.Linear(text_dim, num_tokens * model_dim)
        # Define TransformerDecoder layer with 8 attention heads
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Final MLP to convert processed tokens into point coordinates
        self.mlp = nn.Sequential(
            nn.Linear(num_tokens * model_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )
    
    def forward(self, latent, text_embedding):
        B = latent.size(0)
        # Project latent vector into tokens: (B, num_tokens, model_dim)
        latent_tokens = self.latent_to_tokens(latent).view(B, self.num_tokens, self.model_dim)
        # Project text embedding into tokens: (B, num_tokens, model_dim)
        text_tokens = self.text_to_tokens(text_embedding).view(B, self.num_tokens, self.model_dim)
        # Transformer expects tensor shape: (sequence_length, batch, d_model)
        latent_tokens = latent_tokens.transpose(0, 1)  # (num_tokens, B, model_dim)
        text_tokens = text_tokens.transpose(0, 1)      # (num_tokens, B, model_dim)
        # Cross-attention: latent_tokens (query) attends to text_tokens (memory)
        tokens = self.transformer_decoder(latent_tokens, text_tokens)  # (num_tokens, B, model_dim)
        tokens = tokens.transpose(0, 1).contiguous().view(B, -1)  # (B, num_tokens * model_dim)
        points = self.mlp(tokens)  # (B, num_points * 3)
        points = points.view(B, self.num_points, 3)
        return points

# The full VAE model: uses the Encoder and the new AttentionDecoder
class TextTo3DNet(nn.Module):
    def __init__(self, latent_dim=256, text_dim=512, num_points=1024):
        super(TextTo3DNet, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = AttentionDecoder(latent_dim, text_dim, num_points)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, point_cloud, text_embedding):
        # point_cloud: (B, 4096, 3), text_embedding: (B, 512)
        mu, logvar = self.encoder(point_cloud)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, text_embedding)
        return recon, mu, logvar

def train_text_to_3d(model, dataloader, embedding_generator, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            if batch is None:
                continue
            # collate_fn returns (padded_batch, faces_batch, classes_batch)
            # We need the padded_batch (point clouds) and class names (as text descriptions)
            point_clouds, _, class_names = batch
            point_clouds = point_clouds.to(device)  # (B, 4096, 3)
            batch_size = point_clouds.size(0)
            text_embeddings = []
            for cls in class_names:
                with torch.no_grad():
                    emb = embedding_generator.generate_embedding(cls).to(device)
                    if emb.dim() > 1:
                        emb = emb.squeeze(0)
                    text_embeddings.append(emb)
            text_embeddings = torch.stack(text_embeddings, dim=0)  # (B, 512)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(point_clouds, text_embeddings)
            rec_loss = chamfer_distance(recon, point_clouds)
            kl_loss = kl_divergence(mu, logvar)
            loss = rec_loss + 0.01 * kl_loss  # Adjust the KL weight as needed
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})
        print(f"Epoch {epoch+1} Average Loss: {total_loss/num_batches:.4f}")

def generate_from_text(model, embedding_generator, query_text, device):
    """
    Generates a 3D point cloud based on a text prompt.
    """
    model.eval()
    with torch.no_grad():
        text_emb = embedding_generator.generate_embedding(query_text).to(device)
        if text_emb.dim() > 1:
            text_emb = text_emb.squeeze(0)
        # Add batch dimension to get shape (1, 512)
        text_emb = text_emb.unsqueeze(0)
        # For diversity, choose a random latent vector from a normal distribution
        latent = torch.randn((1, 256)).to(device)
        recon = model.decoder(latent, text_emb)
        # recon: (1, num_points, 3)
        points = recon.squeeze(0).cpu().detach().numpy()
    
    # Create an Open3D point cloud from the generated points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    
    print("[TextTo3D] Performing Poisson surface reconstruction.")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.compute_vertex_normals()
    
    return mesh

def load_existing_model(query, dataset):
    """
    If the query contains a known class name from the dataset,
    return an existing OFF file for that class.
    """
    query_lower = query.lower()
    available_classes = list(dataset.class_to_idx.keys())
    matching_classes = [cls for cls in available_classes if cls.lower() in query_lower]
    if matching_classes:
        target_class = matching_classes[0]
        files = [file_path for file_path, cls in dataset.file_list if cls.lower() == target_class.lower()]
        if files:
            return files[0]  # Return the first matching file (or choose randomly)
    return None

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TextTo3D] Using device: {device}")

    # Initialize the EmbeddingGenerator (ensure spaCy or CLIP models are installed)
    try:
        embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
        print("[TextTo3D] EmbeddingGenerator initialized.")
    except Exception as e:
        print(f"[TextTo3D] Error initializing EmbeddingGenerator: {e}")
        exit(1)

    # Load dataset
    dataset_path = "datasets/CoolDataset"
    try:
        dataset = ModelNet40Dataset(root_dir=dataset_path, split="train")
        print("[TextTo3D] Dataset loaded.")
    except Exception as e:
        print(f"[TextTo3D] Error loading dataset: {e}")
        exit(1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # Initialize our model (generating 1024 points)
    model = TextTo3DNet(latent_dim=256, text_dim=512, num_points=4096).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print("[TextTo3D] Starting training...")
    train_text_to_3d(model, dataloader, embedding_generator, optimizer, device, epochs=10)
    print("[TextTo3D] Training finished.")

    # Main loop for text-to-3D generation
    while True:
        query = input("Enter text for 3D model generation (or 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        # If an existing class is found in the query, load the corresponding existing model
        existing_file = load_existing_model(query, dataset)
        if existing_file:
            print(f"[TextTo3D] Found existing model for query '{query}': {existing_file}")
            mesh = o3d.io.read_triangle_mesh(existing_file)
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], window_name="Existing 3D Model")
        else:
            mesh = generate_from_text(model, embedding_generator, query, device)
            o3d.visualization.draw_geometries([mesh], window_name="Generated 3D Model")