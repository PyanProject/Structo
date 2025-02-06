"""
GAN-based Text-to-3D Generation.

Requirements:
- A dataset of 3D models (ModelNet40Dataset from dataset.py) with OFF files.
- EmbeddingGenerator (from embedding_generator.py) that produces a text embedding of dimension 512.
- gan_model.py which implements Generator and Discriminator.
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
from embedding_generator import EmbeddingGenerator  # Returns text embedding (dim=512)
from gan_model import Generator, Discriminator, train_gan  # GAN components

# Function to perform Poisson reconstruction on the generated point cloud
def reconstruct_mesh_from_points(points):
    """
    Given a numpy array of points (N, 3), creates an Open3D point cloud,
    computes normals, and performs Poisson surface reconstruction to obtain a mesh.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    
    print("[GAN Text-to-3D] Performing Poisson surface reconstruction.")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.compute_vertex_normals()
    
    return mesh

# Function for text-based 3D generation via GAN
def generate_from_text_gan(generator, embedding_generator, query_text, device, noise_dim=100):
    """
    Generates a 3D point cloud using the GAN generator given a text prompt.
    """
    generator.eval()
    with torch.no_grad():
        # Generate text embedding for the query
        text_emb = embedding_generator.generate_embedding(query_text).to(device)
        if text_emb.dim() > 1:
            text_emb = text_emb.squeeze(0)
        # Expand to batch dimension: (1, 512)
        text_emb = text_emb.unsqueeze(0)
        # Generate random noise vector: (1, noise_dim)
        noise = torch.randn((1, noise_dim)).to(device)
        # Generate point cloud using the GAN generator: (1, 4096, 3)
        gen_points = generator(noise, text_emb)
        points = gen_points.squeeze(0).cpu().detach().numpy()
    mesh = reconstruct_mesh_from_points(points)
    return mesh

# Function to retrieve existing model from dataset based on text query
def load_existing_model(query, dataset):
    """
    If the query contains a class name from the dataset,
    returns the corresponding OFF file path.
    """
    query_lower = query.lower()
    available_classes = list(dataset.class_to_idx.keys())
    matching_classes = [cls for cls in available_classes if cls.lower() in query_lower]
    if matching_classes:
        target_class = matching_classes[0]
        files = [file_path for file_path, cls in dataset.file_list if cls.lower() == target_class.lower()]
        if files:
            return files[0]
    return None

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GAN Text-to-3D] Using device: {device}")

    # Initialize EmbeddingGenerator
    try:
        embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
        print("[GAN Text-to-3D] EmbeddingGenerator initialized.")
    except Exception as e:
        print(f"[GAN Text-to-3D] Error initializing EmbeddingGenerator: {e}")
        exit(1)

    # Load dataset
    dataset_path = "datasets/CoolDataset"
    try:
        dataset = ModelNet40Dataset(root_dir=dataset_path, split="train")
        print("[GAN Text-to-3D] Dataset loaded.")
    except Exception as e:
        print(f"[GAN Text-to-3D] Error loading dataset: {e}")
        exit(1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # Initialize GAN components: Generator and Discriminator
    noise_dim = 100
    generator = Generator(noise_dim=noise_dim, embedding_dim=512).to(device)
    discriminator = Discriminator(data_dim=4096 * 3, embedding_dim=512).to(device)
    
    # Train GAN model
    gan_epochs = 10  # Adjust number of epochs as needed
    lr = 0.0001
    print("[GAN Text-to-3D] Starting GAN training...")
    train_gan(generator, discriminator, dataloader, embedding_generator, epochs=gan_epochs, lr=lr, device=device)
    print("[GAN Text-to-3D] GAN training finished.")

    # Main loop for text-to-3D generation
    while True:
        query = input("Enter text for 3D model generation (or 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        
        # If an existing model is found in the query, perform retrieval
        existing_file = load_existing_model(query, dataset)
        if existing_file:
            print(f"[GAN Text-to-3D] Found existing model for query '{query}': {existing_file}")
            mesh = o3d.io.read_triangle_mesh(existing_file)
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], window_name="Existing 3D Model")
        else:
            # Generate model using GAN
            mesh = generate_from_text_gan(generator, embedding_generator, query, device, noise_dim=noise_dim)
            o3d.visualization.draw_geometries([mesh], window_name="Generated 3D Model")