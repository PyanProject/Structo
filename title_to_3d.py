"""
Text-to-3D Generation via Stable Diffusion (text→image) and NeRF (image→3D) Pipeline.

Requirements:
- diffusers (for Stable Diffusion)
- Open3D, OpenCV, torch, numpy, scikit-image, scipy
"""

import os
import torch
import numpy as np
import open3d as o3d
import cv2
from skimage.measure import marching_cubes
import math
import scipy.ndimage
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# Function to reconstruct a 3D mesh from an image-derived volume.
# ------------------------------------------------------------------------------
def reconstruct_mesh_from_volume(image, depth=64, threshold=0.5):
    # Convert input image to grayscale if it is in color.
    if len(image.shape) == 3:
        depth_map = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        depth_map = image.copy()
        
    # Normalize depth map to [0, 255]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map = depth_map * 255.0

    # Convert depth map to normalized values in [0,1]
    depth_normalized = depth_map / 255.0  # shape: (H, W)
    height, width = depth_normalized.shape

    # Create a 3D volume from the depth map.
    z = np.linspace(0, 1, depth)[:, None, None]  # shape: (depth, 1, 1)
    volume = (z <= depth_normalized[None, :, :]).astype(np.float32)

    # Apply Gaussian smoothing to reduce artifacts.
    volume = scipy.ndimage.gaussian_filter(volume, sigma=1)

    # Extract surface mesh using marching cubes.
    verts, faces, normals, values = marching_cubes(volume, level=threshold)

    # Normalize vertices to the unit cube.
    verts[:, 0] = verts[:, 0] / (depth - 1)
    verts[:, 1] = verts[:, 1] / (height - 1)
    verts[:, 2] = verts[:, 2] / (width - 1)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    return mesh

# ------------------------------------------------------------------------------
# Function to generate an image using Stable Diffusion.
# ------------------------------------------------------------------------------
def generate_neural_image_from_text(prompt, width=256, height=256):
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        raise ImportError("Stable Diffusion pipeline is not installed. Please install diffusers and dependencies.")

    model_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(
         model_id, 
         torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Optionally disable the safety checker.
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
         pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # Generate image using Stable Diffusion.
    result = pipe(prompt, width=width, height=height, num_inference_steps=100, guidance_scale=8.5)
    image = result.images[0]
    # Force explicit conversion of PIL image to a numpy array 
    image = np.asarray(image)
    return image

######################################################################
## Main block: Text-to-Image-to-3D model generation and visualization.
######################################################################
if __name__ == "__main__":
    print("[Text-Image-3D] Starting pipeline using Stable Diffusion for image generation, MiDaS for depth estimation, and Poisson reconstruction for mesh extraction.")
    while True:
         query = input("Enter prompt for 3D model generation (or 'exit' to quit): ")
         if query.lower() in ["exit", "quit"]:
             break

         # Generate sensor image and 3D mesh.
         image, mesh = generate_model_from_text(query)

         # Visualize the generated 3D mesh.
         o3d.visualization.draw_geometries([mesh], window_name="Generated 3D Model")

######################################################################
## Helper Functions for Text-to-Image-to-3D Pipeline
######################################################################

def refine_depth_map(depth_map):
    """
    Apply bilateral filtering to the depth map to smooth the noise while preserving edges.
    The depth_map is assumed to be normalized in [0,1].
    """
    depth_map_uint8 = np.uint8(255 * depth_map)
    refined = cv2.bilateralFilter(depth_map_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    refined = refined.astype(np.float32) / 255.0
    return refined

def depth_to_point_cloud(depth_map, color_image, scale=3.0):
    if not isinstance(color_image, np.ndarray):
         color_image = np.array(color_image)
    h, w = depth_map.shape
    depth = depth_map * scale
    cx, cy = w / 2.0, h / 2.0
    f = w
    points = []
    colors = []
    for v in range(h):
         for u in range(w):
              z = depth[v, u]
              if z <= 0:
                  continue
              x = (u - cx) * z / f
              y = (v - cy) * z / f
              points.append([x, y, z])
              colors.append(color_image[v, u, :] / 255.0)
    return np.array(points), np.array(colors)

def create_mesh_from_point_cloud_poisson(points, colors, depth=8):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.01))
    return mesh

def reconstruct_mesh_from_depth(color_image):
    depth_map = estimate_depth(color_image)
    depth_map = refine_depth_map(depth_map)
    points, colors = depth_to_point_cloud(depth_map, color_image)
    mesh = create_mesh_from_point_cloud_poisson(points, colors)
    mesh.compute_vertex_normals()
    return mesh

def generate_model_from_text(prompt, image_size=256):
    base_image = generate_neural_image_from_text(prompt, width=image_size, height=image_size)
    output_dir = "generated_images"
    if not os.path.exists(output_dir):
         os.makedirs(output_dir)
    seed = int.from_bytes(prompt.encode(), 'little') % (2 ** 32)
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt).strip().replace(" ", "_")
    image_filename = os.path.join(output_dir, f"{safe_prompt[:30]}_{seed}.png")
    cv2.imwrite(image_filename, base_image)
    print(f"Generated image saved to {image_filename}")
    mesh = reconstruct_mesh_from_depth(base_image)
    return base_image, mesh 