#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import trimesh


def chamfer_distance(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    reduce_mean: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Вычисление расстояния Чамфера между двумя наборами точек.
    
    Args:
        pred_points: Предсказанные точки формы [batch_size, num_points, 3].
        target_points: Целевые точки формы [batch_size, num_points, 3].
        reduce_mean: Если True, возвращает среднее расстояние по батчу,
                     иначе возвращает кортеж (dist_pred_to_target, dist_target_to_pred).
    
    Returns:
        Среднее расстояние Чамфера или кортеж индивидуальных расстояний.
    """
    # Получаем размеры батча и количество точек
    batch_size, num_points_pred, _ = pred_points.shape
    _, num_points_target, _ = target_points.shape
    
    # Вычисляем попарные расстояния между точками
    # Расширяем размерности для вычисления попарных расстояний
    pred_points_expanded = pred_points.unsqueeze(2)  # [batch_size, num_points_pred, 1, 3]
    target_points_expanded = target_points.unsqueeze(1)  # [batch_size, 1, num_points_target, 3]
    
    # Вычисляем квадрат евклидова расстояния
    distances = torch.sum((pred_points_expanded - target_points_expanded) ** 2, dim=-1)  # [batch_size, num_points_pred, num_points_target]
    
    # Находим минимальные расстояния
    dist_pred_to_target = torch.min(distances, dim=2)[0]  # [batch_size, num_points_pred]
    dist_target_to_pred = torch.min(distances, dim=1)[0]  # [batch_size, num_points_target]
    
    # Усредняем по точкам
    dist_pred_to_target = torch.mean(dist_pred_to_target, dim=1)  # [batch_size]
    dist_target_to_pred = torch.mean(dist_target_to_pred, dim=1)  # [batch_size]
    
    if reduce_mean:
        # Возвращаем среднее значение в обоих направлениях
        return torch.mean(dist_pred_to_target + dist_target_to_pred) / 2.0
    else:
        # Возвращаем индивидуальные расстояния
        return dist_pred_to_target, dist_target_to_pred


def hausdorff_distance(
    pred_points: torch.Tensor,
    target_points: torch.Tensor
) -> torch.Tensor:
    """
    Вычисление расстояния Хаусдорфа между двумя наборами точек.
    
    Args:
        pred_points: Предсказанные точки формы [batch_size, num_points, 3].
        target_points: Целевые точки формы [batch_size, num_points, 3].
    
    Returns:
        Максимальное направленное расстояние Хаусдорфа формы [batch_size].
    """
    # Получаем размеры батча и количество точек
    batch_size, num_points_pred, _ = pred_points.shape
    _, num_points_target, _ = target_points.shape
    
    # Вычисляем попарные расстояния между точками (аналогично расстоянию Чамфера)
    pred_points_expanded = pred_points.unsqueeze(2)  # [batch_size, num_points_pred, 1, 3]
    target_points_expanded = target_points.unsqueeze(1)  # [batch_size, 1, num_points_target, 3]
    
    # Вычисляем квадрат евклидова расстояния
    distances = torch.sum((pred_points_expanded - target_points_expanded) ** 2, dim=-1)  # [batch_size, num_points_pred, num_points_target]
    
    # Находим минимальные расстояния для каждой точки
    min_dist_pred_to_target = torch.min(distances, dim=2)[0]  # [batch_size, num_points_pred]
    min_dist_target_to_pred = torch.min(distances, dim=1)[0]  # [batch_size, num_points_target]
    
    # Находим максимальное из минимальных расстояний (направленное расстояние Хаусдорфа)
    directed_hausdorff_pred_to_target = torch.max(min_dist_pred_to_target, dim=1)[0]  # [batch_size]
    directed_hausdorff_target_to_pred = torch.max(min_dist_target_to_pred, dim=1)[0]  # [batch_size]
    
    # Возвращаем максимальное из двух направленных расстояний
    hausdorff = torch.max(directed_hausdorff_pred_to_target, directed_hausdorff_target_to_pred)
    
    return torch.sqrt(hausdorff)  # Берем корень, чтобы получить настоящее расстояние, а не его квадрат


def iou_3d(
    pred_voxels: torch.Tensor,
    target_voxels: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Вычисление IoU (Intersection over Union) для 3D воксельных сеток.
    
    Args:
        pred_voxels: Предсказанные вокселы формы [batch_size, 1, depth, height, width].
        target_voxels: Целевые вокселы формы [batch_size, 1, depth, height, width].
        threshold: Порог для бинаризации вокселей.
    
    Returns:
        IoU для каждого элемента батча формы [batch_size].
    """
    # Бинаризуем вокселы
    pred_binary = (pred_voxels > threshold).float()
    target_binary = (target_voxels > threshold).float()
    
    # Вычисляем пересечение и объединение
    intersection = torch.sum(pred_binary * target_binary, dim=[1, 2, 3, 4])
    union = torch.sum(torch.clamp(pred_binary + target_binary, 0, 1), dim=[1, 2, 3, 4])
    
    # Вычисляем IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def fscore(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    threshold: float = 0.01
) -> torch.Tensor:
    """
    Вычисление F-score для наборов точек.
    
    Args:
        pred_points: Предсказанные точки формы [batch_size, num_points, 3].
        target_points: Целевые точки формы [batch_size, num_points, 3].
        threshold: Порог расстояния для определения соответствия точек.
    
    Returns:
        F-score для каждого элемента батча формы [batch_size].
    """
    # Вычисляем расстояния Чамфера (без усреднения по батчу)
    dist_pred_to_target, dist_target_to_pred = chamfer_distance(
        pred_points, target_points, reduce_mean=False
    )
    
    # Вычисляем точность и полноту
    precision = torch.mean((dist_pred_to_target < threshold).float(), dim=0)
    recall = torch.mean((dist_target_to_pred < threshold).float(), dim=0)
    
    # Вычисляем F-score
    f_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    return f_score


def normal_consistency(
    pred_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    num_samples: int = 10000
) -> float:
    """
    Вычисление согласованности нормалей между двумя мешами.
    
    Args:
        pred_mesh: Предсказанный меш.
        target_mesh: Целевой меш.
        num_samples: Количество точек для выборки с поверхности.
    
    Returns:
        Средняя согласованность нормалей.
    """
    # Сэмплируем точки и нормали с поверхности
    pred_points, pred_face_idx = pred_mesh.sample(num_samples, return_index=True)
    target_points, target_face_idx = target_mesh.sample(num_samples, return_index=True)
    
    # Получаем нормали для сэмплированных точек
    pred_normals = pred_mesh.face_normals[pred_face_idx]
    target_normals = target_mesh.face_normals[target_face_idx]
    
    # Конвертируем в тензоры PyTorch
    pred_points = torch.tensor(pred_points, dtype=torch.float32)
    target_points = torch.tensor(target_points, dtype=torch.float32)
    pred_normals = torch.tensor(pred_normals, dtype=torch.float32)
    target_normals = torch.tensor(target_normals, dtype=torch.float32)
    
    # Для каждой точки из pred находим ближайшую точку из target
    distances = torch.cdist(pred_points, target_points)
    nearest_idx = torch.argmin(distances, dim=1)
    
    # Получаем нормали ближайших точек
    nearest_target_normals = target_normals[nearest_idx]
    
    # Вычисляем косинус угла между нормалями (скалярное произведение нормализованных векторов)
    pred_normals_normalized = pred_normals / torch.norm(pred_normals, dim=1, keepdim=True)
    nearest_target_normals_normalized = nearest_target_normals / torch.norm(nearest_target_normals, dim=1, keepdim=True)
    
    cosine_similarity = torch.sum(pred_normals_normalized * nearest_target_normals_normalized, dim=1)
    
    # Вычисляем абсолютное значение косинуса (нормаль может быть направлена в противоположную сторону)
    cosine_similarity = torch.abs(cosine_similarity)
    
    # Возвращаем средний косинус
    return torch.mean(cosine_similarity).item()


def mesh_accuracy(
    pred_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    num_samples: int = 10000,
    threshold: float = 0.01
) -> Dict[str, float]:
    """
    Вычисление комплексной метрики точности для 3D-мешей.
    
    Args:
        pred_mesh: Предсказанный меш.
        target_mesh: Целевой меш.
        num_samples: Количество точек для выборки с поверхности.
        threshold: Порог расстояния для определения соответствия точек.
    
    Returns:
        Словарь с метриками:
            - chamfer: Расстояние Чамфера.
            - hausdorff: Расстояние Хаусдорфа.
            - f_score: F-score.
            - normal_consistency: Согласованность нормалей.
    """
    # Сэмплируем точки с поверхности
    pred_points = torch.tensor(pred_mesh.sample(num_samples), dtype=torch.float32).unsqueeze(0)
    target_points = torch.tensor(target_mesh.sample(num_samples), dtype=torch.float32).unsqueeze(0)
    
    # Вычисляем метрики
    chamfer = chamfer_distance(pred_points, target_points).item()
    hausdorff = hausdorff_distance(pred_points, target_points).item()
    f_score_val = fscore(pred_points, target_points, threshold).item()
    normal_cons = normal_consistency(pred_mesh, target_mesh, num_samples)
    
    return {
        'chamfer': chamfer,
        'hausdorff': hausdorff,
        'f_score': f_score_val,
        'normal_consistency': normal_cons
    }


def clip_score(
    text_features: torch.Tensor,
    image_features: torch.Tensor
) -> torch.Tensor:
    """
    Вычисление CLIP-score между текстовыми и визуальными фичами.
    
    Args:
        text_features: Текстовые фичи формы [batch_size, feature_dim].
        image_features: Изображения фичи формы [batch_size, feature_dim].
    
    Returns:
        CLIP-score формы [batch_size].
    """
    # Нормализуем фичи
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    
    # Вычисляем сходство
    similarity = torch.sum(text_features * image_features, dim=1)
    
    return similarity 