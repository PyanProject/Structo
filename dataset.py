import os


class TemporaryDataset:
    """
    Генерация временного датасета для тренировок.
    """
    def __init__(self, output_dir="temp_dataset"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_dataset(self, num_samples=50):
        samples = []
        for i in range(num_samples):
            filepath = os.path.join(self.output_dir, f"sample_{i}.ply")
            with open(filepath, "w") as f:
                f.write("placeholder content")  # Заглушка для файла 3D модели
            samples.append({"id": i, "filepath": filepath})
        print(f"Создано {num_samples} образцов в {self.output_dir}")
        return samples
