import numpy as np
import torch

def compute_iou(pred_voxels, target_voxels, threshold=0.5):
    """
    Calculates the IoU (Intersection over Union) metric for voxel grids.
    
    Args:
        pred_voxels (torch.Tensor): Predicted voxel grid.
        target_voxels (torch.Tensor): Target voxel grid.
        threshold (float): Binarization threshold.
        
    Returns:
        float: IoU value.
    """
    # Convert to numpy if needed
    if isinstance(pred_voxels, torch.Tensor):
        pred_voxels = pred_voxels.detach().cpu().numpy()
    if isinstance(target_voxels, torch.Tensor):
        target_voxels = target_voxels.detach().cpu().numpy()
    
    # Binarization
    pred_binary = (pred_voxels > threshold).astype(np.float32)
    target_binary = (target_voxels > threshold).astype(np.float32)
    
    # Calculate IoU
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
        
    iou = intersection / union
    return float(iou)

def compute_chamfer_distance(pred_points, target_points):
    """
    Calculates the Chamfer Distance metric between point clouds.
    
    Args:
        pred_points (numpy.ndarray): Predicted point cloud.
        target_points (numpy.ndarray): Target point cloud.
        
    Returns:
        float: Chamfer Distance value.
    """
    # Check that both point sets are not empty
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    
    # Convert to numpy if needed
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.detach().cpu().numpy()
    if isinstance(target_points, torch.Tensor):
        target_points = target_points.detach().cpu().numpy()
    
    # Calculate distance matrix
    pred_points_exp = np.expand_dims(pred_points, axis=1)  # [N, 1, 3]
    target_points_exp = np.expand_dims(target_points, axis=0)  # [1, M, 3]
    dist_matrix = np.sqrt(np.sum((pred_points_exp - target_points_exp) ** 2, axis=2))  # [N, M]
    
    # Minimum distances for each point
    pred_to_target = np.min(dist_matrix, axis=1)  # [N]
    target_to_pred = np.min(dist_matrix, axis=0)  # [M]
    
    # Chamfer Distance - average distance
    chamfer_dist = np.mean(pred_to_target) + np.mean(target_to_pred)
    
    return float(chamfer_dist)

def compute_f_score(pred_voxels, target_voxels, threshold=0.5, d_threshold=0.01):
    """
    Calculates F-score for evaluating 3D shape reproduction accuracy.
    
    Args:
        pred_voxels (torch.Tensor): Predicted voxel grid.
        target_voxels (torch.Tensor): Target voxel grid.
        threshold (float): Binarization threshold.
        d_threshold (float): Distance threshold for F-score.
        
    Returns:
        float: F-score value.
    """
    # Convert to numpy if needed
    if isinstance(pred_voxels, torch.Tensor):
        pred_voxels = pred_voxels.detach().cpu().numpy()
    if isinstance(target_voxels, torch.Tensor):
        target_voxels = target_voxels.detach().cpu().numpy()
    
    # Binarization
    pred_binary = (pred_voxels > threshold).astype(np.float32)
    target_binary = (target_voxels > threshold).astype(np.float32)
    
    # Extract coordinates of active voxels
    pred_indices = np.where(pred_binary > 0)
    target_indices = np.where(target_binary > 0)
    
    # Convert indices to coordinates
    if len(pred_indices[0]) == 0 or len(target_indices[0]) == 0:
        return 0.0
        
    pred_points = np.vstack([pred_indices[0], pred_indices[1], pred_indices[2]]).T.astype(np.float32)
    target_points = np.vstack([target_indices[0], target_indices[1], target_indices[2]]).T.astype(np.float32)
    
    # Normalize coordinates
    res = pred_voxels.shape[0]
    pred_points = pred_points / res
    target_points = target_points / res
    
    # Calculate distance matrix
    pred_points_exp = np.expand_dims(pred_points, axis=1)  # [N, 1, 3]
    target_points_exp = np.expand_dims(target_points, axis=0)  # [1, M, 3]
    dist_matrix = np.sqrt(np.sum((pred_points_exp - target_points_exp) ** 2, axis=2))  # [N, M]
    
    # Minimum distances for each point
    pred_to_target = np.min(dist_matrix, axis=1)  # [N]
    target_to_pred = np.min(dist_matrix, axis=0)  # [M]
    
    # Count points closer than the threshold distance
    precision = np.sum(pred_to_target < d_threshold) / len(pred_to_target)
    recall = np.sum(target_to_pred < d_threshold) / len(target_to_pred)
    
    # Avoid division by zero
    if precision + recall == 0:
        return 0.0
    
    # Calculate F-score
    f_score = 2 * precision * recall / (precision + recall)
    
    return float(f_score) 