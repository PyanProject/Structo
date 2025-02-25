import os
import argparse
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """
    Загружает файл по URL с отображением прогресса.
    
    Args:
        url (str): URL для загрузки.
        output_path (str): Путь для сохранения файла.
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_modelnet40(output_dir):
    """
    Загружает и распаковывает датасет ModelNet40.
    
    Args:
        output_dir (str): Директория для сохранения датасета.
    """
    # URL для загрузки датасета
    modelnet40_url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    
    # Создание директории для загрузки, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Путь для сохранения архива
    zip_path = os.path.join(output_dir, "ModelNet40.zip")
    
    # Загрузка архива
    print(f"Загрузка датасета ModelNet40 из {modelnet40_url}...")
    download_url(modelnet40_url, zip_path)
    
    # Распаковка архива
    print(f"Распаковка архива в {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Удаление архива
    os.remove(zip_path)
    
    # Перемещение содержимого в корневую директорию
    extracted_dir = os.path.join(output_dir, "ModelNet40")
    if os.path.exists(extracted_dir):
        # Перемещение всех файлов и директорий
        for item in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, item)
            dst = os.path.join(output_dir, item)
            shutil.move(src, dst)
        
        # Удаление пустой директории
        os.rmdir(extracted_dir)
    
    print(f"Датасет ModelNet40 успешно загружен и распакован в {output_dir}")
    
    # Вывод информации о датасете
    categories = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    print(f"Количество категорий: {len(categories)}")
    print(f"Категории: {', '.join(sorted(categories))}")

def main():
    """
    Основная функция.
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Загрузка датасета ModelNet40")
    parser.add_argument("--output_dir", type=str, default="data/ModelNet40", help="Директория для сохранения датасета")
    args = parser.parse_args()
    
    # Загрузка датасета
    download_modelnet40(args.output_dir)

if __name__ == "__main__":
    main() 