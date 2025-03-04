import os
import argparse
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from shap_e.diffusion.gaussian_diffusion import diffusion_from_config, GaussianDiffusion
from shap_e.models.download import load_model, load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Директория с датасетом")
    parser.add_argument("--output_dir", type=str, default="output", help="Директория для сохранения результатов")
    parser.add_argument("--model_type", type=str, default="text300M", help="Тип модели (text300M или image300M)")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=1e-5, help="Скорость обучения")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Устройство (cuda/cpu)")
    parser.add_argument("--config", type=str, default=None, help="Путь к конфигурационному файлу")
    return parser.parse_args()

def load_dataset(data_dir, batch_size):
    """
    Загрузка датасета. Здесь нужно реализовать загрузку вашего датасета.
    """
    print(f"Загрузка датасета из директории: {data_dir}")
    
    # TODO: Реализуйте загрузку вашего датасета в соответствии с его форматом
    # Пример: создание фиктивного датасета
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            # В реальном датасете здесь должен быть код для загрузки данных
            # Например:
            # self.image_files = glob.glob(os.path.join(data_dir, "images/*.png"))
            # self.point_clouds = glob.glob(os.path.join(data_dir, "points/*.npy"))
            # self.captions = pd.read_csv(os.path.join(data_dir, "captions.csv"))
            
            # Для примера создаем случайные данные
            self.size = 100  # Размер фиктивного датасета
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Здесь должен быть код для загрузки конкретного элемента датасета
            # Например:
            # img = Image.open(self.image_files[idx])
            # points = np.load(self.point_clouds[idx])
            # caption = self.captions.iloc[idx]['caption']
            
            # Для примера возвращаем случайные данные
            batch = {
                'points': torch.randn(3, 16384),  # Облако точек
                'text': 'пример подписи',  # Текстовое описание
            }
            return batch
    
    # Создание датасета и загрузчика
    dataset = DummyDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Датасет успешно загружен: {len(dataset)} примеров")
    return dataloader

def train_model(args):
    """
    Функция обучения модели
    """
    print(f"Начало обучения модели {args.model_type} на устройстве {args.device}")
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем конфигурацию и модель
    print("Загрузка моделей и конфигурации...")
    device = torch.device(args.device)
    
    if args.config:
        # Загрузка пользовательской конфигурации
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        diffusion = diffusion_from_config(config)
    else:
        # Загрузка стандартной конфигурации
        diffusion = diffusion_from_config(load_config('diffusion'))
    
    # Загружаем предобученную модель
    model = load_model(args.model_type, device=device)
    
    # Загружаем датасет
    dataloader = load_dataset(args.data_dir, args.batch_size)
    
    # Настраиваем оптимизатор
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Цикл обучения
    for epoch in range(args.epochs):
        print(f"Эпоха {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader):
            # Перемещаем данные на нужное устройство
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Подготовка данных в зависимости от типа модели
            if args.model_type == "text300M":
                model_kwargs = {'texts': batch.get('text', [''] * args.batch_size)}
            else:  # image300M
                model_kwargs = {'images': batch.get('images')}
            
            # Обучающий шаг
            optimizer.zero_grad()
            
            # Получаем шум и временной шаг
            x_start = batch['points']
            t = torch.randint(0, diffusion.num_timesteps, (args.batch_size,), device=device)
            
            # Вычисляем потери
            losses = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
            loss = losses['loss'].mean()
            
            # Обратное распространение и оптимизация
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Выводим информацию о потерях
        avg_loss = epoch_loss / len(dataloader)
        print(f"Эпоха {epoch+1} завершена. Средняя потеря: {avg_loss:.4f}")
        
        # Сохраняем модель каждую эпоху
        save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Модель сохранена: {save_path}")
    
    # Сохраняем итоговую модель
    final_save_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"Обучение завершено. Итоговая модель сохранена: {final_save_path}")

if __name__ == "__main__":
    args = parse_args()
    train_model(args) 