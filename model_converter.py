#!/usr/bin/env python3
import os
import sys
import tempfile
import shutil
import uuid
from pathlib import Path
import trimesh
import numpy as np
import json

class ModelConverter:
    """Класс для конвертации 3D-моделей из OBJ в другие форматы"""
    
    SUPPORTED_FORMATS = {
        'obj': {'extension': '.obj', 'description': 'Wavefront OBJ'},
        'stl': {'extension': '.stl', 'description': 'STL (Standard Triangle Language)'},
        'ply': {'extension': '.ply', 'description': 'Stanford PLY'},
        'glb': {'extension': '.glb', 'description': 'GLB (Binary glTF)'},
        'dae': {'extension': '.dae', 'description': 'COLLADA'},
        'off': {'extension': '.off', 'description': 'OFF (Object File Format)'}
    }
    
    def __init__(self, models_dir='static/models'):
        """Инициализация конвертера
        
        Args:
            models_dir: Директория с моделями
        """
        self.models_dir = models_dir
        self.temp_dir = tempfile.mkdtemp()
        
        # Создаем директорию для конвертированных моделей, если она не существует
        self.converted_dir = os.path.join(models_dir, 'converted')
        os.makedirs(self.converted_dir, exist_ok=True)
    
    def convert_model(self, model_filename, target_format):
        """Конвертирует модель в указанный формат
        
        Args:
            model_filename: Имя файла модели (без пути)
            target_format: Целевой формат (например, 'stl', 'ply')
            
        Returns:
            dict: Результат конвертации с путем к новому файлу или сообщением об ошибке
        """
        if target_format not in self.SUPPORTED_FORMATS:
            return {'success': False, 'error': f'Неподдерживаемый формат: {target_format}'}
        
        try:
            # Полный путь к исходной модели
            source_path = os.path.join(self.models_dir, model_filename)
            
            # Проверяем существование файла
            if not os.path.exists(source_path):
                return {'success': False, 'error': f'Файл не найден: {source_path}'}
            
            # Создаем имя для конвертированного файла
            base_name = os.path.splitext(model_filename)[0]
            target_extension = self.SUPPORTED_FORMATS[target_format]['extension']
            target_filename = f"{base_name}_{target_format}{target_extension}"
            target_path = os.path.join(self.converted_dir, target_filename)
            
            # Проверяем, существует ли уже конвертированный файл
            if os.path.exists(target_path):
                # Возвращаем путь к существующему файлу
                relative_path = os.path.join('static/models/converted', target_filename)
                return {
                    'success': True, 
                    'filename': target_filename,
                    'path': target_path,
                    'url': relative_path
                }
            
            # Загружаем модель с помощью trimesh
            mesh = trimesh.load(source_path)
            
            # Сохраняем в целевом формате
            if target_format == 'obj':
                # Если исходный формат уже OBJ, просто копируем файл
                shutil.copy(source_path, target_path)
            elif target_format in ['stl', 'ply', 'off', 'glb', 'dae']:
                # Форматы, которые напрямую поддерживаются trimesh
                mesh.export(target_path)
            else:
                return {'success': False, 'error': f'Конвертация в формат {target_format} не реализована'}
            
            # Проверяем, что файл был создан
            if not os.path.exists(target_path):
                return {'success': False, 'error': 'Ошибка при создании конвертированного файла'}
            
            # Возвращаем относительный путь для использования в URL
            relative_path = os.path.join('static/models/converted', target_filename)
            return {
                'success': True, 
                'filename': target_filename,
                'path': target_path,
                'url': relative_path
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    

    
    def get_supported_formats(self):
        """Возвращает список поддерживаемых форматов
        
        Returns:
            list: Список словарей с информацией о форматах
        """
        formats = []
        for format_id, info in self.SUPPORTED_FORMATS.items():
            formats.append({
                'id': format_id,
                'name': info['description'],
                'extension': info['extension']
            })
        return formats
    
    def cleanup(self):
        """Очищает временные файлы"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

# Если скрипт запущен напрямую, выполняем тестовую конвертацию
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python model_converter.py <имя_файла.obj> <формат>")
        sys.exit(1)
    
    model_filename = sys.argv[1]
    target_format = sys.argv[2]
    
    converter = ModelConverter()
    result = converter.convert_model(model_filename, target_format)
    
    print(json.dumps(result, indent=2))
    converter.cleanup() 