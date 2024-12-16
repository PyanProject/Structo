import os
import trimesh
import numpy as np
from torch.utils.data import Dataset
import random

class CustomDataset:
    def __init__(self, output_dir="temp_dataset"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Синонимы для цветов (включая варианты без ё)
        colors_variants = {
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

        colors_map = {
            "красная": [1.0, 0.0, 0.0],
            "синяя": [0.0, 0.0, 1.0],
            "зелёная": [0.0, 1.0, 0.0],
            "жёлтая": [1.0, 1.0, 0.0],
            "фиолетовая": [1.0, 0.0, 1.0],
            "оранжевая": [1.0, 0.5, 0.0],
            "белая": [1.0, 1.0, 1.0],
            "черная": [0.0, 0.0, 0.0]
        }

        # Синонимы для форм
        shapes_variants = {
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

        shapes_map = [
            ("сфера", "sphere"),
            ("куб", "cube"),
            ("конус", "cone"),
            ("цилиндр", "cylinder"),
            ("пирамида", "pyramid"),
            ("тор", "torus"),
            ("цилиндрическая труба", "cylindrical_tube")
        ]

        self.data = []
        for i in range(100):
            # Выбираем базовый цвет и форму
            base_color_name = list(colors_map.keys())[i % len(colors_map)]
            base_color = colors_map[base_color_name]
            color_variant = random.choice(colors_variants[base_color_name])

            base_shape_name, shape_type = shapes_map[i % len(shapes_map)]
            shape_variant = random.choice(shapes_variants[base_shape_name])

            size_param = round(np.random.uniform(0.5, 3.0), 2)
            # Формируем текст с вариативными синонимами
            text = f"{color_variant} {shape_variant} с радиусом {size_param}"
            self.data.append({"text": text, "model": shape_type, "size_param": size_param, "color": base_color})
    
    def generate_dataset(self):
        samples = []
        for idx, sample in enumerate(self.data):
            model_type = sample["model"]
            size_param = sample["size_param"]

            if model_type == "sphere":
                mesh = trimesh.creation.icosphere(radius=size_param)
            elif model_type == "cube":
                mesh = trimesh.creation.box(extents=[size_param, size_param, size_param])
            elif model_type == "cone":
                mesh = trimesh.creation.cone(radius=size_param, height=size_param*1.5)
            elif model_type == "cylinder":
                mesh = trimesh.creation.cylinder(radius=size_param, height=size_param*2)
            elif model_type == "pyramid":
                mesh = self.create_pyramid([size_param, size_param, size_param*1.5])
            elif model_type == "torus":
                mesh = trimesh.creation.torus(major_radius=size_param, minor_radius=size_param*0.3)
            elif model_type == "cylindrical_tube":
                mesh = trimesh.creation.cylinder(radius=size_param*0.5, height=size_param*2)
            else:
                print(f"[DATASET] Неизвестный тип модели: {model_type}")
                continue

            color_val = sample["color"]
            vertex_colors = np.tile((np.array(color_val)*255).astype(np.uint8), (len(mesh.vertices), 1))
            alpha_channel = np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)
            vertex_colors = np.hstack((vertex_colors, alpha_channel))
            mesh.visual.vertex_colors = vertex_colors

            filepath = os.path.join(self.output_dir, f"sample_{idx}.ply")
            mesh.export(filepath)
            samples.append({"text": sample["text"], "filepath": filepath})
            print(f"[DATASET] {idx}: {sample['text']} -> {filepath}")

        print(f"[DATASET] Создано {len(samples)} образцов")
        return samples

    def create_pyramid(self, extents):
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

class Temporary3DDataset(Dataset):
    def __init__(self, samples, embedding_generator):
        self.samples = samples
        self.embedding_generator = embedding_generator

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        print(f"[TEMP DATASET] Генерация эмбеддинга для: {text}")
        embedding = self.embedding_generator.generate_embedding(text).squeeze()
        filepath = sample["filepath"]
        return embedding, filepath
