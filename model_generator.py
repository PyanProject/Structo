'''
этот файл раньше помогал дегенерировать простые модели, а сейчас его нужно переделать под датасет

'''


import numpy as np
import trimesh
import os
import hashlib
from embedding_generator import EnsureScalar

# при генерации модели она сохраняется и ей присваивается уникальное имя. очень важная ф-ция
def generate_unique_filename(text: str, output_dir: str) -> str:
    hash_object = hashlib.md5(text.encode())
    filename = f"model_{hash_object.hexdigest()[:8]}.ply"
    return os.path.join(output_dir, filename)

# не дает файлам переполниться, ф-ция-мусорщик
def manage_model_files(output_dir: str, max_files: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(output_dir) if f.endswith('.ply')]
    if len(files) > max_files:
        oldest_file = min(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
        os.remove(os.path.join(output_dir, oldest_file))
        print(f"[MODEL GEN] Удалён старый файл: {oldest_file}")

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# срал
def create_pyramid(extents=[1.0, 1.0, 2.0]):
    width, depth, height = extents
    vertices = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, depth, 0],
        [0, depth, 0],
        [width / 2, depth / 2, height]
    ])
    faces = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 1, 2],
        [0, 2, 3]
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def extract_color_and_shape(text: str):
    # Добавим нормализацию текста: привести к нижнему регистру
    text_lower = text.lower()

    # Цвета и их синонимы
    color_synonyms = {
        "красная": [
            "красная", "красн.", "красн", "красный", "краснаяя", "красныйя", 
            "red", "redd", "redy", "read"
        ],

        "синяя": [
            "синяя", "синий", "син.", "син", "синийя", "синяяя", 
            "blue", "blu", "bloo", "bleu"
        ],

        "зелёная": [
            "зеленая", "зелёная", "зелен.", "зелен", "зел.", "зел",
            "зелёный", "зеленый", "зеленаяя", "зелёнаяя", 
            "green", "grin", "gren", "grean"
        ],

        "жёлтая": [
            "желтая", "жёлтая", "желт.", "желт", "жёлт.", "жёлт", 
            "жёлтый", "желтый", "желтаяя", "жёлтаяя", 
            "yellow", "yelow", "yello", "yeallow"
        ],

        "фиолетовая": [
            "фиолетовая", "фиолет.", "фиолет", "фиолетовый", "фиолетоваяя", 
            "violet", "viollet", "violett", "viloet", "vilot"
        ],

        "оранжевая": [
            "оранжевая", "оранж.", "оранж", "оранжеваяя", "оранжевый", 
            "orange", "orng", "orang", "orrange", "ornage"
        ],

        "белая": [
            "белая", "бел.", "бел", "белаяя", "белый", 
            "white", "wite", "whte", "wihte", "whitte"
        ],

        "черная": [
            "черная", "чёрная", "черн.", "черн", "чёрн.", "чёрн", 
            "черный", "чёрный", "чернаяя", "чёрнаяя", 
            "black", "blak", "blck", "balck", "blac"
        ]
    }

    color_map = {
        "красная": [1.0, 0.0, 0.0],
        "синяя": [0.0, 0.0, 1.0],
        "зелёная": [0.0, 1.0, 0.0],
        "жёлтая": [1.0, 1.0, 0.0],
        "фиолетовая": [1.0, 0.0, 1.0],
        "оранжевая": [1.0, 0.5, 0.0],
        "белая": [1.0, 1.0, 1.0],
        "черная": [0.0, 0.0, 0.0]
    }

    # Формы и синонимы
    shape_synonyms = {
        "сфера": [
            "сфера", "сф.", "шар", "сферическая", "сферичный", 
            "sphere", "sfer", "sphera", "sphare", "шарик"
        ],

        "куб": [
            "куб", "кубик", "кубический", "кубик.", "кубич.", 
            "cube", "kub", "kubik", "qbe", "cub"
        ],
        
        "конус": [
            "конус", "кон.", "конический", "конусообразный", 
            "cone", "konus", "kon", "coan", "con", "конусик"
        ],

        "цилиндр": [
            "цилиндр", "цил.", "цилин.", "цилиндрический", 
            "cylinder", "cilindr", "zilindr", "cilinder", "sylinder", "цилиндрик"
        ],

        "пирамида": [
            "пирамида", "пир.", "пир", "пирамидальная", "пирамид.", 
            "pyramid", "piramid", "pyrmid", "pirameed", "piramed", "пирамидка"
        ],

        "тор": [
            "тор", "кольцо", "тороид", "торообразный", 
            "torus", "tor", "tore", "thorus", "tohr"
        ],

        "цилиндрическая труба": [
            "цилиндрическая труба", "цил. труба", "труба", "трубка", 
            "cylindrical tube", "tube", "tubular cylinder", "pipe", "cylinder pipe", "трубочка"
        ]
    }

    shape_map = {
        "сфера": "sphere",
        "куб": "cube",
        "конус": "cone",
        "цилиндр": "cylinder",
        "пирамида": "pyramid",
        "тор": "torus",
        "цилиндрическая труба": "cylindrical_tube"
    }

    found_color = None
    found_shape = None
    size_param = 1.0

    words = text_lower.split()
    # Поиск размера
    for w in words:
        w_clean = w.replace(',', '.')
        if w_clean.replace('.', '', 1).isdigit():
            try:
                size_param = float(w_clean)
            except:
                pass

    # Поиск цвета
    for base_color, synonyms in color_synonyms.items():
        for syn in synonyms:
            if syn in words:
                found_color = base_color
                break
        if found_color is not None:
            break

    # Поиск формы
    for base_shape, synonyms in shape_synonyms.items():
        for syn in synonyms:
            if syn in words:
                found_shape = base_shape
                break
        if found_shape is not None:
            break

    if found_color is None:
        found_color = "белая"
    if found_shape is None:
        found_shape = "сфера"

    color_val = color_map[found_color]
    shape_val = shape_map[found_shape]

    return shape_val, color_val, size_param

def generate_3d_scene_from_embedding(embedding: np.ndarray, text: str, output_dir: str = "models") -> str:
    print("[MODEL GEN] Генерация сцены...")
    shape, color, requested_size = extract_color_and_shape(text)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding type: {type(embedding)}")

    embedding_normalized = (embedding - embedding.mean()) / (embedding.std() + 1e-8)
    print(f"Embedding normalized shape: {embedding_normalized.shape}")

    shape_param = embedding_normalized[0, 0]
    size_param = embedding_normalized[0, 1]

    print(f"Shape param: {shape_param}, type: {type(shape_param)}")
    print(f"Size param: {size_param}, type: {type(size_param)}")

    shape_param = normalize(shape_param, -1, 1)
    final_size = np.clip((requested_size + size_param), 0.5, 5.0)
    
    print(f"Final size before conversion: {final_size}, type: {type(final_size)}")

    final_size = float(final_size)
    print(f"Final size after conversion: {final_size}, type: {type(final_size)}")
    
    final_size = EnsureScalar(final_size)
    final_color = np.array(color)

    if final_size <= 0:
        raise ValueError("final_size must be a positive number.")

    # Create the mesh with the scalar final_size
    mesh = trimesh.creation.icosphere(radius=final_size)

    if shape == "sphere":
        mesh = trimesh.creation.icosphere(radius=final_size)
    elif shape == "cube":
        mesh = trimesh.creation.box(extents=[final_size, final_size, final_size])
    elif shape == "cone":
        mesh = trimesh.creation.cone(radius=final_size, height=final_size*1.5)
    elif shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=final_size, height=final_size*2)
    elif shape == "pyramid":
        mesh = create_pyramid([final_size, final_size, final_size*1.5])
    elif shape == "torus":
        mesh = trimesh.creation.torus(major_radius=final_size, minor_radius=final_size*0.3)
    elif shape == "cylindrical_tube":
        mesh = trimesh.creation.cylinder(radius=final_size*0.5, height=final_size*2)
    else:
        mesh = trimesh.creation.icosphere(radius=final_size)

    vertex_colors = np.tile((final_color*255).astype(np.uint8), (len(mesh.vertices), 1))
    alpha_channel = np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)
    vertex_colors = np.hstack((vertex_colors, alpha_channel))
    mesh.visual.vertex_colors = vertex_colors

    print(f"[MODEL GEN] Форма: {shape}, размер: {final_size}, цвет: {final_color}, текст: '{text}'")
    manage_model_files(output_dir)
    scene_filename = generate_unique_filename(text, output_dir)
    mesh.export(scene_filename)
    print(f"[MODEL GEN] Модель сохранена: {scene_filename}")
    return scene_filename
