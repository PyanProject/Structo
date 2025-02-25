import numpy as np
import trimesh
import trimesh.voxel as vox
from scipy import ndimage

def mesh_to_voxel(mesh, resolution=64, padding=0.05):
    """
    Преобразует 3D меш в воксельное представление.
    
    Args:
        mesh (trimesh.Trimesh): 3D меш для преобразования.
        resolution (int): Разрешение воксельной сетки.
        padding (float): Дополнительный отступ для воксельной сетки.
        
    Returns:
        numpy.ndarray: Воксельное представление меша размером [resolution, resolution, resolution].
    """
    # Убедимся, что меш центрирован и нормализован по размеру
    mesh.vertices -= mesh.bounding_box.centroid
    max_dim = np.max(mesh.bounding_box.extents)
    scale = (1.0 - 2 * padding) / max_dim
    mesh.vertices *= scale
    
    # Альтернативный метод вокселизации
    try:
        # Пробуем использовать встроенную функцию trimesh
        if hasattr(vox.VoxelGrid, 'from_mesh'):
            voxels = vox.VoxelGrid.from_mesh(mesh, pitch=2.0/resolution, method='ray')
            voxel_array = voxels.matrix.astype(float)
        elif hasattr(trimesh, 'voxel'):
            # Альтернативный метод в новых версиях trimesh
            voxels = trimesh.voxel.creation.voxelize(mesh, resolution)
            voxel_array = voxels.matrix.astype(float)
        else:
            # Используем простую растеризацию, если специализированные методы недоступны
            bounds = mesh.bounds
            voxel_size = (bounds[1] - bounds[0]).max() / resolution
            voxel_array = np.zeros((resolution, resolution, resolution), dtype=float)
            
            # Заполняем воксели, содержащие часть меша
            for face in mesh.triangles:
                # Определяем bounds лица
                face_min = np.min(face, axis=0)
                face_max = np.max(face, axis=0)
                
                # Вычисляем индексы воксельной сетки
                min_x = max(0, int((face_min[0] - bounds[0][0]) / voxel_size))
                min_y = max(0, int((face_min[1] - bounds[0][1]) / voxel_size))
                min_z = max(0, int((face_min[2] - bounds[0][2]) / voxel_size))
                max_x = min(resolution-1, int((face_max[0] - bounds[0][0]) / voxel_size))
                max_y = min(resolution-1, int((face_max[1] - bounds[0][1]) / voxel_size))
                max_z = min(resolution-1, int((face_max[2] - bounds[0][2]) / voxel_size))
                
                # Помечаем воксели
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        for z in range(min_z, max_z + 1):
                            # Примитивная проверка - может быть улучшена
                            voxel_array[x, y, z] = 1.0
                            
            # Заполняем пустоты с помощью морфологических операций
            voxel_array = ndimage.binary_fill_holes(voxel_array).astype(float)
    except Exception as e:
        print(f"Ошибка при вокселизации: {e}")
        # Создаем простую вокселизацию как запасной вариант
        voxel_array = np.zeros((resolution, resolution, resolution), dtype=float)
        # Заполняем центральный воксель
        center = resolution // 2
        size = resolution // 4
        voxel_array[center-size:center+size, center-size:center+size, center-size:center+size] = 1.0
    
    # Проверка размерности и ресайз при необходимости
    if voxel_array.shape[0] != resolution or voxel_array.shape[1] != resolution or voxel_array.shape[2] != resolution:
        # Добавляем нули, если размерность меньше необходимой
        padded_array = np.zeros((resolution, resolution, resolution), dtype=float)
        
        # Копируем существующие данные
        min_x = min(voxel_array.shape[0], resolution)
        min_y = min(voxel_array.shape[1], resolution)
        min_z = min(voxel_array.shape[2], resolution)
        
        padded_array[:min_x, :min_y, :min_z] = voxel_array[:min_x, :min_y, :min_z]
        voxel_array = padded_array
    
    return voxel_array

def voxel_to_mesh(voxels, threshold=0.5):
    """
    Преобразует воксельное представление в 3D меш.
    
    Args:
        voxels (numpy.ndarray): Воксельное представление.
        threshold (float): Порог бинаризации воксельной сетки.
        
    Returns:
        trimesh.Trimesh: 3D меш.
    """
    # Бинаризация воксельной сетки
    binary_voxels = voxels > threshold
    
    try:
        # Создание меша из воксельной сетки - адаптация для разных версий trimesh
        if hasattr(vox, 'VoxelGrid'):
            # Для старых версий
            voxel_grid = vox.VoxelGrid(binary_voxels)
            if hasattr(voxel_grid, 'as_mesh'):
                mesh = voxel_grid.as_mesh()
            else:
                # Альтернативный вариант
                from trimesh.voxel.encoding import DenseEncoding
                from trimesh.voxel.ops import matrix_to_marching_cubes
                encoding = DenseEncoding(binary_voxels)
                mesh = matrix_to_marching_cubes(encoding.matrix)
        elif hasattr(trimesh.voxel, 'creation'):
            # Для новых версий
            voxel_grid = trimesh.voxel.VoxelGrid(binary_voxels)
            mesh = voxel_grid.as_mesh()
        else:
            # Запасной вариант - создаем простой куб
            mesh = trimesh.creation.box()
            print("Используется запасной вариант для создания меша - простой куб")
    except Exception as e:
        print(f"Ошибка при создании меша из воксельной сетки: {e}")
        # Создаем простой куб как запасной вариант
        mesh = trimesh.creation.box()
    
    # Нормализация размера меша
    mesh.vertices -= mesh.bounding_box.centroid
    max_dim = np.max(mesh.bounding_box.extents)
    if max_dim > 0:
        mesh.vertices /= max_dim
    
    return mesh

def save_mesh(mesh, filepath):
    """
    Сохраняет 3D меш в файл.
    
    Args:
        mesh (trimesh.Trimesh): 3D меш для сохранения.
        filepath (str): Путь для сохранения файла.
    """
    # Определение формата файла на основе расширения
    file_extension = filepath.split('.')[-1].lower()
    
    # Экспорт меша в заданном формате
    if file_extension == 'obj':
        mesh.export(filepath, file_type='obj')
    elif file_extension == 'stl':
        mesh.export(filepath, file_type='stl')
    elif file_extension == 'ply':
        mesh.export(filepath, file_type='ply')
    else:
        # По умолчанию сохраняем в формате OBJ
        mesh.export(filepath + '.obj', file_type='obj')

def process_voxel_prediction(voxel_grid, threshold=0.5, smooth=True, sigma=0.5):
    """
    Обрабатывает предсказание воксельной сетки для получения качественного меша.
    
    Args:
        voxel_grid (numpy.ndarray): Воксельное представление.
        threshold (float): Порог бинаризации воксельной сетки.
        smooth (bool): Применять ли сглаживание.
        sigma (float): Параметр сглаживания Гаусса.
        
    Returns:
        trimesh.Trimesh: Обработанный 3D меш.
    """
    # Применяем сглаживание по Гауссу, если требуется
    if smooth:
        voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=sigma)
    
    # Бинаризация воксельной сетки
    binary_voxels = voxel_grid > threshold
    
    # Заполняем дыры
    binary_voxels = ndimage.binary_fill_holes(binary_voxels)
    
    # Удаляем маленькие компоненты (шум)
    labeled_array, num_features = ndimage.label(binary_voxels)
    if num_features > 1:
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # Игнорируем фон
        largest_component = np.argmax(component_sizes)
        binary_voxels = labeled_array == largest_component
    
    # Используем нашу безопасную функцию конвертации
    mesh = voxel_to_mesh(binary_voxels, threshold=0.5)
    
    # Убедимся, что меш не пустой
    if mesh.is_empty:
        return None
    
    return mesh 