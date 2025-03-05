import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import trimesh
from scipy.spatial.distance import directed_hausdorff

def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    reduce_mean: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Вычисление расстояния Чамфера между двумя наборами точек.
    
    Args:
        x: Первый набор точек (B, N, 3)
        y: Второй набор точек (B, M, 3)
        reduce_mean: Усреднять ли результат по батчу
        
    Returns:
        Расстояние Чамфера или кортеж (расстояние x->y, расстояние y->x)
    """
    # Вычисляем попарные расстояния
    xx = torch.sum(x ** 2, dim=2, keepdim=True)     # (B, N, 1)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)     # (B, M, 1)
    xy = torch.matmul(x, y.transpose(1, 2))         # (B, N, M)
    
    dist = xx + yy.transpose(1, 2) - 2 * xy         # (B, N, M)
    
    # Находим минимальные расстояния
    dist_x = torch.min(dist, dim=2)[0]              # (B, N)
    dist_y = torch.min(dist, dim=1)[0]              # (B, M)
    
    if reduce_mean:
        dist_x = torch.mean(dist_x, dim=1)          # (B,)
        dist_y = torch.mean(dist_y, dim=1)          # (B,)
        return torch.mean(dist_x + dist_y)
    else:
        return dist_x, dist_y

def hausdorff_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Вычисление расстояния Хаусдорфа между двумя наборами точек.
    
    Args:
        x: Первый набор точек (N, 3)
        y: Второй набор точек (M, 3)
        
    Returns:
        Расстояние Хаусдорфа
    """
    return max(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0])

def iou_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Вычисление IoU для 3D воксельных сеток.
    
    Args:
        pred: Предсказанная воксельная сетка (B, D, H, W)
        target: Целевая воксельная сетка (B, D, H, W)
        threshold: Порог для бинаризации
        
    Returns:
        IoU для каждого элемента в батче
    """
    # Бинаризация
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    # Вычисление пересечения и объединения
    intersection = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum(pred + target - pred * target, dim=(1, 2, 3))
    
    # Избегаем деления на ноль
    union = torch.clamp(union, min=1e-6)
    
    return intersection / union

def fscore(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Вычисление F-score для наборов точек.
    
    Args:
        pred_points: Предсказанные точки (B, N, 3)
        target_points: Целевые точки (B, M, 3)
        threshold: Порог расстояния
        
    Returns:
        F-score, точность и полнота
    """
    # Вычисляем расстояния Чамфера
    dist_pred, dist_target = chamfer_distance(pred_points, target_points, reduce_mean=False)
    
    # Вычисляем точность и полноту
    precision = torch.mean((dist_pred < threshold).float(), dim=1)
    recall = torch.mean((dist_target < threshold).float(), dim=1)
    
    # Вычисляем F-score
    precision = torch.clamp(precision, min=1e-6)
    recall = torch.clamp(recall, min=1e-6)
    f_score = 2 * precision * recall / (precision + recall)
    
    return f_score, precision, recall

def normal_consistency(
    pred_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh
) -> float:
    """
    Вычисление согласованности нормалей между мешами.
    
    Args:
        pred_mesh: Предсказанный меш
        target_mesh: Целевой меш
        
    Returns:
        Значение согласованности нормалей
    """
    # Получаем нормали вершин
    pred_normals = pred_mesh.vertex_normals
    target_normals = target_mesh.vertex_normals
    
    # Нормализуем нормали
    pred_normals = pred_normals / np.linalg.norm(pred_normals, axis=1, keepdims=True)
    target_normals = target_normals / np.linalg.norm(target_normals, axis=1, keepdims=True)
    
    # Вычисляем скалярное произведение
    consistency = np.abs(np.sum(pred_normals * target_normals, axis=1))
    
    return np.mean(consistency)

def mesh_accuracy(
    pred_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    num_samples: int = 10000
) -> Dict[str, float]:
    """
    Вычисление комплексной метрики точности меша.
    
    Args:
        pred_mesh: Предсказанный меш
        target_mesh: Целевой меш
        num_samples: Количество точек для сэмплирования
        
    Returns:
        Словарь с метриками
    """
    # Сэмплируем точки
    pred_points = pred_mesh.sample(num_samples)
    target_points = target_mesh.sample(num_samples)
    
    # Переводим в тензоры
    pred_points = torch.from_numpy(pred_points).unsqueeze(0).float()
    target_points = torch.from_numpy(target_points).unsqueeze(0).float()
    
    # Вычисляем метрики
    chamfer = chamfer_distance(pred_points, target_points).item()
    hausdorff = hausdorff_distance(pred_points[0].numpy(), target_points[0].numpy())
    f_score, precision, recall = fscore(pred_points, target_points)
    normal_cons = normal_consistency(pred_mesh, target_mesh)
    
    return {
        'chamfer_distance': chamfer,
        'hausdorff_distance': hausdorff,
        'f_score': f_score.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'normal_consistency': normal_cons
    }

def clip_score(
    text_features: torch.Tensor,
    image_features: torch.Tensor
) -> torch.Tensor:
    """
    Вычисление CLIP-score между текстом и изображениями.
    
    Args:
        text_features: Текстовые признаки (B, D)
        image_features: Признаки изображений (B, N, D)
        
    Returns:
        CLIP-score для каждого элемента в батче
    """
    # Нормализация признаков
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)
    
    # Вычисляем схожесть
    similarity = torch.matmul(text_features, image_features.transpose(-2, -1))
    
    # Берем максимальное значение по ракурсам
    scores = torch.max(similarity, dim=-1)[0]
    
    return scores 