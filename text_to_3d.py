"""
Text-to-3D Generation via Text → Image → 3D Pipeline.

Requirements:
- A dataset of 3D models (ModelNet40Dataset from dataset.py) with OFF files.
- EmbeddingGenerator (from embedding_generator.py) that produces a text embedding of dimension 512.
- This file implements a text→image→3D pipeline.
"""

import os
import torch
import numpy as np
import open3d as o3d
import requests  # For external image retrieval via HTTP
from tqdm import tqdm
from opensimplex import OpenSimplex  # Using opensimplex for 2D noise generation

import cv2  # Used for image processing
from dataset import ModelNet40Dataset, collate_fn
from embedding_generator import EmbeddingGenerator  # (if needed for retrieval or further conditioning)
from skimage.measure import marching_cubes  # For isosurface extraction from volumetric data
import math

# ------------------------------------------------------------------------------
# Function to generate an image from text.
# The image is generated using OpenSimplex noise with fBm,
# then enhanced via sigmoid mapping for increased contrast.
# ------------------------------------------------------------------------------
def generate_image_from_text(query_text, embedding_generator=None):
    """
    Generates an image (numpy array of shape (256, 256, 3)) from a text prompt using OpenSimplex noise.
    Deterministic output based on the hash of the query for consistency.
    If embedding_generator is provided, it modulates contrast and brightness.
    """
    # Если запрос начинается с "neural:", используем нейросетевую генерацию изображения.
    if query_text.lower().startswith("neural:"):
         prompt = query_text[len("neural:"):].strip()
         try:
              image = generate_neural_image_from_text(prompt)
              print("Image generated using neural network.")
              return image
         except Exception as e:
              print(f"Neural generation failed: {e}. Falling back to noise-based generation.")

    # Seed for noise generation (fallback to noise-based generation)
    seed = int.from_bytes(query_text.encode(), 'little') % (2 ** 32)
    generator = OpenSimplex(seed)

    # Noise parameters (can be modulated via embedding if available)
    scale_factor = 20.0   # Lower scale factor => finer details
    octaves = 4
    persistence = 0.6
    lacunarity = 2.0

    # Optionally modulate noise parameters with text embedding:
    if embedding_generator is not None:
        embedding = embedding_generator.generate_embedding(query_text)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        # Adjust scale factor based on embedding mean (example modulation)
        scale_factor = 20.0 * (0.9 + (np.mean(embedding) % 0.2))
        # Similarly, brightness offset can be computed if needed.
    
    # Generate noise image using fBm
    noise_img = np.zeros((256, 256), dtype=np.float32)
    
    def fBm(x, y, octaves, persistence, lacunarity, generator):
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        total_amplitude = 0.0
        for _ in range(octaves):
            value += generator.noise2(x * frequency, y * frequency) * amplitude
            total_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return value / total_amplitude

    for i in range(256):
        for j in range(256):
            noise_img[i, j] = fBm(i / scale_factor, j / scale_factor, octaves, persistence, lacunarity, generator)
    
    # Normalize noise to [0,1]
    noise_min = noise_img.min()
    noise_max = noise_img.max()
    noise_norm = (noise_img - noise_min) / (noise_max - noise_min)

    # Apply a sigmoid mapping to enhance contrast.
    # A high alpha (e.g. 10) yields a near binary transition.
    alpha = 10.0
    sigmoid_mapped = 1.0 / (1.0 + np.exp(-alpha * (noise_norm - 0.5)))
    image = (sigmoid_mapped * 255).astype(np.uint8)

    # Optionally adjust brightness/contrast if embedding is provided.
    if embedding_generator is not None:
        modifier = 0.8 + (np.mean(embedding) % 0.4)  # e.g. in range [0.8, 1.2]
        brightness = (np.std(embedding) % 30)          # brightness offset
        image = np.clip(image * modifier + brightness, 0, 255).astype(np.uint8)
    
    # For consistency, replicate grayscale to 3 channels.
    image = np.stack([image, image, image], axis=-1)
    
    # Apply histogram equalization for further contrast enhancement.
    gray_eq = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    image = np.stack([gray_eq, gray_eq, gray_eq], axis=-1)
    
    # Final light Gaussian blur to remove minor noise.
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image

# ------------------------------------------------------------------------------
# New function: Create a volumetric reconstruction and extract mesh using marching cubes.
# The volume is built by applying a sigmoid to the normalized grayscale image,
# which yields a sharper transition between filled and unfilled voxels.
# ------------------------------------------------------------------------------
def reconstruct_mesh_from_volume(image, depth=64, threshold=0.5):
    # Преобразование входного изображения в grayscale для получения карты глубины.
    if len(image.shape) == 3:
        depth_map = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        depth_map = image.copy()
        
    # Normalize depth map to the [0, 255] range.
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map = depth_map * 255.0

    # Convert depth map to normalized [0, 1] values.
    depth_normalized = depth_map / 255.0  # shape: (H, W)
    height, width = depth_normalized.shape

    # Create a 3D volume from the depth map.
    # The volume will have 'depth' slices along the z-axis.
    # For each pixel, we fill voxels below the estimated normalized depth value.
    z = np.linspace(0, 1, depth)[:, None, None]  # shape: (depth, 1, 1)
    volume = (z <= depth_normalized[None, :, :]).astype(np.float32)

    # Apply Gaussian smoothing to reduce artifacts in the volume.
    import scipy.ndimage
    volume = scipy.ndimage.gaussian_filter(volume, sigma=1)

    # Extract surface mesh using marching cubes from the volume.
    verts, faces, normals, values = marching_cubes(volume, level=threshold)

    # Adjust vertices to a normalized coordinate system (unit cube).
    verts[:, 0] = verts[:, 0] / (depth - 1)
    verts[:, 1] = verts[:, 1] / (height - 1)
    verts[:, 2] = verts[:, 2] / (width - 1)

    # Create and return an Open3D triangle mesh.
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    return mesh

# ------------------------------------------------------------------------------
# Main pipeline: Choose between NeRF generation and legacy text→image→3D pipeline.
# ------------------------------------------------------------------------------
def generate_model_from_text(query_text, embedding_generator=None):
    """
    If query starts with 'nerf:', generate a NeRF-based scene.
    Otherwise, fallback to the original text→image→3D pipeline with primitives support.
    """
    lower_query = query_text.lower()
    if lower_query.startswith("nerf:"):
         # Remove the "nerf:" prefix.
         prompt = query_text[len("nerf:"):].strip()
         return generate_nerf_from_text(prompt, embedding_generator)
    elif lower_query.startswith("neural:"):
         # Use neural image generation and reconstruct 3D model using NeRF.
         return generate_nerf_from_neural_image(query_text, embedding_generator)
    # Generate specific primitives if possible.
    if any(keyword in prompt_lower for keyword in ["cube", "box", "cuboid", "rectangular"]):
         mesh = o3d.geometry.TriangleMesh.create_box()
         mesh.compute_vertex_normals()
         return mesh
    elif "sphere" in prompt_lower:
         mesh = o3d.geometry.TriangleMesh.create_sphere()
         mesh.compute_vertex_normals()
         return mesh
    elif "cylinder" in prompt_lower:
         mesh = o3d.geometry.TriangleMesh.create_cylinder()
         mesh.compute_vertex_normals()
         return mesh
    elif "cone" in prompt_lower:
         mesh = o3d.geometry.TriangleMesh.create_cone()
         mesh.compute_vertex_normals()
         return mesh
    else:
         # Generate noise-based image (legacy pipeline).
         image = generate_image_from_text(query_text, embedding_generator)
         
         # Save the generated image for inspection.
         seed = int.from_bytes(query_text.encode(), 'little') % (2 ** 32)
         output_dir = "generated_images"
         if not os.path.exists(output_dir):
             os.makedirs(output_dir)
         safe_query = "".join(c for c in query_text if c.isalnum() or c in (' ', '_')).strip().replace(" ", "_")
         image_filename = os.path.join(output_dir, f"{safe_query[:30]}_{seed}.png")
         cv2.imwrite(image_filename, image)
         print(f"Generated image saved to {image_filename}")
         
         # Reconstruct 3D mesh from the image.
         mesh = reconstruct_mesh_from_volume(image)
         return mesh

# ------------------------------------------------------------------------------
# Function for retrieving an existing model based on text query.
# If the query contains a class name from the dataset, return the corresponding OFF file path.
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# New function: Generate an image using a neural network (Stable Diffusion).
# ------------------------------------------------------------------------------
def generate_neural_image_from_text(prompt, width=256, height=256):
    try:
        from diffusers import StableDiffusionPipeline
        import torch
    except ImportError as e:
        raise ImportError("Stable Diffusion pipeline is not installed. Please install diffusers and dependencies.")

    model_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(
         model_id, 
         torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Optionally disable the safety checker to prevent excessive filtering
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
         pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # Use additional parameters for more detailed generation
    result = pipe(prompt, width=width, height=height, num_inference_steps=50, guidance_scale=7.5)
    image = np.array(result.images[0])
    return image

# ------------------------------------------------------------------------------
# New function: Generate a NeRF scene from text.
# This simplified version simulates a NeRF pipeline by generating multiple views from the text prompt,
# aggregating them to form a pseudo depth map, and reconstructing a mesh as a placeholder for a true NeRF.
# ------------------------------------------------------------------------------
def generate_nerf_from_text(query_text, embedding_generator=None, num_views=5):
    """
    Pipeline for generating a Neural Radiance Field (NeRF) scene from a text prompt.
    This is a simplified version simulating NeRF generation.
    """
    views = []
    seed = int.from_bytes(query_text.encode(), 'little') % (2 ** 32)
    for i in range(num_views):
         # Calculate viewing angle in degrees.
         angle = 360 * i / num_views
         # Create a modified prompt for the current view.
         view_prompt = f"{query_text} view angle {angle:.1f}"
         # Generate image using existing text-to-image function.
         view_image = generate_image_from_text(view_prompt, embedding_generator)
         # Simulate view variation by rotating the image.
         center = (view_image.shape[1] // 2, view_image.shape[0] // 2)
         M = cv2.getRotationMatrix2D(center, angle, 1.0)
         rotated_view = cv2.warpAffine(view_image, M, (view_image.shape[1], view_image.shape[0]))
         views.append(rotated_view)
    # Aggregate generated views to simulate a depth map (placeholder for NeRF volume rendering).
    avg_view = np.mean(np.stack(views, axis=0), axis=0).astype(np.uint8)
    # Reconstruct a mesh from the aggregated view.
    nerf_mesh = reconstruct_mesh_from_volume(avg_view)
    return nerf_mesh

# ------------------------------------------------------------------------------
# New function: Generate a 3D model using NeRF from a neural-generated image.
# It uses a neural network (Stable Diffusion) to generate a base image,
# then simulates multiple views by rotating the image, aggregates them to form a pseudo depth map,
# and reconstructs a 3D mesh as a placeholder for a true NeRF.
# ------------------------------------------------------------------------------
def generate_nerf_from_neural_image(query_text, embedding_generator=None, num_views=5):
    """
    Pipeline for generating a Neural Radiance Field (NeRF) 3D model from a neural-generated image.
    It first generates a base image using the neural network and then creates multiple rotated views.
    """
    # Remove "neural:" prefix.
    base_prompt = query_text[len("neural:"):].strip()
    # Generate base image using neural generation.
    base_image = generate_neural_image_from_text(base_prompt)
    views = []
    for i in range(num_views):
         # Calculate viewing angle in degrees.
         angle = 360 * i / num_views
         center = (base_image.shape[1] // 2, base_image.shape[0] // 2)
         M = cv2.getRotationMatrix2D(center, angle, 1.0)
         rotated_view = cv2.warpAffine(base_image, M, (base_image.shape[1], base_image.shape[0]))
         views.append(rotated_view)
    # Aggregate generated views to simulate a depth map.
    avg_view = np.mean(np.stack(views, axis=0), axis=0).astype(np.uint8)
    # Reconstruct a mesh from the aggregated view.
    nerf_mesh = reconstruct_mesh_from_volume(avg_view)
    return nerf_mesh

# ------------------------------------------------------------------------------
# Main block: Initialization, optional retrieval, and generation of the 3D model.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set device (not used in current pipeline, but may be useful if embedding conditioning used)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Text-Image-3D] Using device: {device}")

    # Initialize EmbeddingGenerator if available.
    try:
        embedding_generator = EmbeddingGenerator(device, reduced_dim=512)
        print("[Text-Image-3D] EmbeddingGenerator initialized.")
    except Exception as e:
        print(f"[Text-Image-3D] Error initializing EmbeddingGenerator: {e}")
        embedding_generator = None

    # Load dataset for retrieval (if applicable)
    dataset_path = "datasets/CoolDataset"
    try:
        dataset = ModelNet40Dataset(root_dir=dataset_path, split="train")
        print("[Text-Image-3D] Dataset loaded.")
    except Exception as e:
        print(f"[Text-Image-3D] Error loading dataset: {e}")
        dataset = None

    # Main loop: prompt for text and generate a 3D model/scene.
    while True:
        query = input("Enter text for NeRF scene generation (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break

        # Generate model using the appropriate pipeline.
        mesh = generate_model_from_text(query, embedding_generator)
        # Display the resulting 3D model.
        if query.lower().startswith("nerf:"):
             o3d.visualization.draw_geometries([mesh], window_name="Generated NeRF Scene")
        else:
             o3d.visualization.draw_geometries([mesh], window_name="Generated 3D Model")