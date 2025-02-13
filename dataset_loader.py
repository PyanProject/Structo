# dataset_loader.py

import os
import sys
import urllib.request
import zipfile
from tqdm import tqdm

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None

def select_local_dataset():
    """
    Позволяет выбрать локальную директорию с датасетом через диалоговое окно Qt.
    """
    if QtWidgets is None:
        print("PyQt5 не установлен. Локальный выбор датасета невозможен.")
        return None
    app = QtWidgets.QApplication(sys.argv)
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    directory = QtWidgets.QFileDialog.getExistingDirectory(
        None, "Выберите директорию с датасетом", os.getcwd(), options=options
    )
    if directory:
        return directory
    return None

# Расширенный словарь известных датасетов.
# Обратите внимание: некоторые ссылки являются примерами и могут требовать корректировки.
DATASET_LINKS = {
    "chairs": "http://example.com/datasets/chairs.zip",            # заменить на рабочую ссылку
    "tables": "http://example.com/datasets/tables.zip",              # заменить на рабочую ссылку
    "modelnet40": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
    "modelnet10": "http://modelnet.cs.princeton.edu/ModelNet10.zip",
    "shapenetcore": "http://shapenet.cs.stanford.edu/ShapeNetCore.v1.zip",
    "shapenetsem": "http://shapenet.cs.stanford.edu/ShapeNetSem.v1.zip",
}   

def download_progress_hook(t):
    last_b = [0]
    def update_to(block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            t.total = total_size
        downloaded = block_num * block_size - last_b[0]
        t.update(downloaded)
        last_b[0] = block_num * block_size
    return update_to

def download_dataset(url, target_dir):
    """
    Загружает датасет по указанной ссылке, отображая прогресс с помощью tqdm,
    и распаковывает его в target_dir.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    zip_path = os.path.join(target_dir, "dataset.zip")
    print(f"Загружаем датасет из {url} ...")
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=zip_path, reporthook=download_progress_hook(t))
    print("Загрузка завершена. Распаковка архива...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(zip_path)
    print(f"Датасет распакован в {target_dir}")
    return target_dir

def load_dataset(source, download_path="datasets"):
    """
    Принимает source, который может быть:
      - Локальным путём (строкой) к датасету,
      - Известным именем датасета (например, 'chairs', 'modelnet40') из DATASET_LINKS,
      - Или списком таких значений (для гибридного варианта).
    
    Если указан download_path, датасеты будут скачаны в эту директорию.
    Возвращает список директорий с датасетами.
    """
    datasets = []
    if isinstance(source, list):
        sources = source
    else:
        sources = [source]

    for src in sources:
        # Если передан локальный путь
        if os.path.exists(src):
            datasets.append(src)
        # Если передано известное имя датасета (регистр не важен)
        elif src.lower() in DATASET_LINKS:
            url = DATASET_LINKS[src.lower()]
            dataset_dir = os.path.join(download_path, src.lower())
            if not os.path.exists(dataset_dir):
                dataset_dir = download_dataset(url, dataset_dir)
            datasets.append(dataset_dir)
        else:
            print(f"Источник датасета '{src}' не найден как локальный путь или известное имя.")
    return datasets

def list_available_datasets():
    """
    Возвращает список доступных имен датасетов из DATASET_LINKS.
    """
    return list(DATASET_LINKS.keys())

def clear_dataset_cache(download_path="datasets"):
    """
    Удаляет все скачанные датасеты из указанного каталога.
    """
    if os.path.exists(download_path):
        for item in os.listdir(download_path):
            item_path = os.path.join(download_path, item)
            if os.path.isdir(item_path):
                try:
                    import shutil
                    shutil.rmtree(item_path)
                    print(f"Удалена директория: {item_path}")
                except Exception as e:
                    print(f"Ошибка при удалении {item_path}: {e}")
    else:
        print(f"Путь {download_path} не существует.")

if __name__ == "__main__":
    print("Доступные датасеты:")
    for ds in list_available_datasets():
        print(f"  - {ds}")
    print("\nВыберите режим загрузки датасета:")
    print("1. Локальный выбор (через Qt)")
    print("2. Загрузка из известных датасетов")
    print("3. Очистка кэша скачанных датасетов")
    choice = input("Введите 1, 2 или 3: ").strip()
    if choice == "1":
        ds = select_local_dataset()
        if ds:
            print("Выбран локальный датасет:", ds)
        else:
            print("Датасет не выбран.")
    elif choice == "2":
        ds_name = input("Введите имя датасета (например, modelnet40): ").strip()
        ds_list = load_dataset(ds_name)
        if ds_list:
            print("Датасет(ы) загружен(ы):", ds_list)
        else:
            print("Датасет не найден или не загружен.")
    elif choice == "3":
        clear_dataset_cache()
    else:
        print("Неверный выбор.")
