import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import random
import sys

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.voxelization import mesh_to_voxel

class ModelNetDataset(Dataset):
    """
    Датасет для работы с ModelNet40.
    """
    
    def __init__(self, root_dir, split='train', transform=None, voxel_resolution=64, text_augmentation=True):
        """
        Инициализация датасета ModelNet40.
        
        Args:
            root_dir (str): Путь к директории с датасетом.
            split (str): Разделение датасета ('train' или 'test').
            transform (callable, optional): Трансформации данных.
            voxel_resolution (int): Разрешение воксельной сетки.
            text_augmentation (bool): Использовать ли аугментацию текста.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.voxel_resolution = voxel_resolution
        self.text_augmentation = text_augmentation
        
        assert self.split in ['train', 'test'], "Split должен быть 'train' или 'test'"
        
        # Загрузка списка классов
        self.categories = sorted([d for d in os.listdir(root_dir) 
                                 if os.path.isdir(os.path.join(root_dir, d))])
        
        # Словарь для преобразования названий классов в описания на естественном языке
        self.class_descriptions = {
            "airplane": ["самолет", "воздушное судно", "летающий аппарат"],
            "bathtub": ["ванна", "джакузи", "купель для ванной"],
            "bed": ["кровать", "спальное место", "ложе"],
            "bench": ["скамейка", "лавка", "парковая скамья"],
            "bookshelf": ["книжная полка", "стеллаж для книг", "шкаф для книг"],
            "bottle": ["бутылка", "сосуд для жидкости", "емкость для напитков"],
            "bowl": ["миска", "чаша", "пиала"],
            "car": ["автомобиль", "машина", "легковой транспорт"],
            "chair": ["стул", "кресло", "сиденье"],
            "cone": ["конус", "коническая форма", "конусообразный предмет"],
            "cup": ["чашка", "кружка", "стакан"],
            "curtain": ["занавеска", "штора", "портьера"],
            "desk": ["стол", "письменный стол", "рабочий стол"],
            "door": ["дверь", "дверная панель", "вход"],
            "dresser": ["комод", "шкаф", "мебель для хранения"],
            "flower_pot": ["цветочный горшок", "горшок для растений", "кашпо"],
            "glass_box": ["стеклянная коробка", "аквариум", "стеклянный куб"],
            "guitar": ["гитара", "акустический инструмент", "струнный инструмент"],
            "keyboard": ["клавиатура", "устройство ввода", "компьютерная периферия"],
            "lamp": ["лампа", "светильник", "осветительный прибор"],
            "laptop": ["ноутбук", "портативный компьютер", "лэптоп"],
            "mantel": ["камин", "портал камина", "каминная полка"],
            "monitor": ["монитор", "экран", "дисплей"],
            "night_stand": ["тумбочка", "прикроватная тумба", "ночной столик"],
            "person": ["человек", "фигура человека", "манекен"],
            "piano": ["пианино", "фортепиано", "клавишный инструмент"],
            "plant": ["растение", "комнатное растение", "декоративный цветок"],
            "radio": ["радио", "радиоприемник", "аудиоустройство"],
            "range_hood": ["вытяжка", "кухонная вытяжка", "очиститель воздуха"],
            "sink": ["раковина", "мойка", "умывальник"],
            "sofa": ["диван", "софа", "мягкая мебель"],
            "stairs": ["лестница", "ступени", "лестничный пролет"],
            "stool": ["табурет", "табуретка", "низкое сиденье"],
            "table": ["стол", "обеденный стол", "столешница"],
            "tent": ["палатка", "тент", "временное укрытие"],
            "toilet": ["унитаз", "туалет", "сантехника"],
            "tv_stand": ["тумба под телевизор", "подставка для ТВ", "медиа-консоль"],
            "vase": ["ваза", "декоративный сосуд", "емкость для цветов"],
            "wardrobe": ["шкаф", "гардероб", "платяной шкаф"],
            "xbox": ["игровая консоль", "приставка", "игровая система"]
        }
        
        # Шаблоны для аугментации текста
        self.text_templates = [
            "трехмерный объект: {}",
            "3D модель объекта, называемого {}",
            "трехмерная модель {}",
            "{} в трехмерном пространстве",
            "компьютерная 3D модель объекта: {}",
            "визуализация трехмерного предмета: {}",
            "3D рендер объекта: {}",
            "объемная модель {}"
        ]
        
        # Атрибуты описания
        self.attributes = [
            "простой", "сложный", "детализированный", "геометрический", 
            "обычный", "современный", "минималистичный", "классический",
            "стандартный", "базовый", "функциональный", "практичный"
        ]
        
        # Загрузка списка файлов
        self.file_list = []
        for category in self.categories:
            category_dir = os.path.join(root_dir, category, self.split)
            
            if not os.path.exists(category_dir):
                continue
                
            files = sorted(os.listdir(category_dir))
            off_files = [f for f in files if f.endswith('.off')]
            
            for off_file in off_files:
                self.file_list.append({
                    'path': os.path.join(category_dir, off_file),
                    'category': category
                })
                
        print(f"Загружено {len(self.file_list)} файлов для {split} набора")
    
    def __len__(self):
        """
        Возвращает размер датасета.
        
        Returns:
            int: Количество образцов в датасете.
        """
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Получает элемент датасета по индексу.
        
        Args:
            idx (int): Индекс элемента.
            
        Returns:
            dict: Словарь с данными образца.
        """
        file_info = self.file_list[idx]
        file_path = file_info['path']
        category = file_info['category']
        
        # Загрузка меша
        try:
            mesh = trimesh.load(file_path)
            
            # Нормализация размера и позиции
            mesh.vertices -= mesh.bounding_box.centroid
            max_dim = np.max(mesh.bounding_box.extents)
            mesh.vertices /= max_dim
            
            # Преобразование в воксельное представление
            voxels = mesh_to_voxel(mesh, resolution=self.voxel_resolution)
            
            # Преобразование в тензор
            voxel_tensor = torch.FloatTensor(voxels).unsqueeze(0)  # [1, D, H, W]
            
            # Генерация текстового описания
            text_prompt = self.generate_text_prompt(category)
            
            # Если нужны преобразования, применяем их
            if self.transform is not None:
                voxel_tensor = self.transform(voxel_tensor)
            
            return {
                'voxels': voxel_tensor,
                'text': text_prompt,
                'category': category,
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            # Возвращаем первый элемент в случае ошибки
            return self.__getitem__(0) if idx != 0 else None
    
    def generate_text_prompt(self, category):
        """
        Генерирует текстовое описание для категории.
        
        Args:
            category (str): Категория объекта.
            
        Returns:
            str: Текстовое описание.
        """
        # Базовое описание без аугментации
        if not self.text_augmentation or self.split == 'test':
            return f"3D модель объекта: {category}"
        
        # Выбор случайного шаблона
        template = random.choice(self.text_templates)
        
        # Выбор случайного описания для категории
        if category in self.class_descriptions:
            description = random.choice(self.class_descriptions[category])
        else:
            description = category
        
        # Добавляем случайные атрибуты (с вероятностью 70%)
        if random.random() < 0.7:
            attribute = random.choice(self.attributes)
            description = f"{attribute} {description}"
        
        # Формируем финальный текст
        text_prompt = template.format(description)
        
        return text_prompt 