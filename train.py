import os
import argparse
import torch
import yaml
from tqdm import tqdm
import logging
import gc
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import platform

from shap_e.diffusion.gaussian_diffusion import diffusion_from_config, GaussianDiffusion
from shap_e.models.download import load_model, load_config
from src.objaverse_dataset import get_objaverse_dataloader, prepare_objaverse_data, MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_rocm_available():
    """Проверка доступности ROCm"""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None

def get_device_info():
    """Получение информации об устройстве"""
    if torch.cuda.is_available():
        return "CUDA", torch.cuda.get_device_name(0)
    elif is_rocm_available():
        return "ROCm", "AMD GPU"
    return "CPU", platform.processor()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Директория с датасетом")
    parser.add_argument("--output_dir", type=str, default="output", help="Директория для сохранения результатов")
    parser.add_argument("--model_type", type=str, default="text300M", help="Тип модели (text300M или image300M)")
    parser.add_argument("--batch_size", type=int, default=2, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=1e-4, help="Скорость обучения")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Устройство (cuda/cpu)")
    parser.add_argument("--config", type=str, default=None, help="Путь к конфигурационному файлу")
    parser.add_argument("--dataset", type=str, default="dummy", choices=["dummy", "objaverse"],
                        help="Тип датасета для обучения")
    parser.add_argument("--prepare_data", action="store_true",
                        help="Подготовить данные Objaverse перед обучением")
    parser.add_argument("--num_objects", type=int, default=5000,
                        help="Количество объектов для загрузки из Objaverse")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Количество шагов для накопления градиента")
    parser.add_argument("--max_points", type=int, default=4096,
                        help="Максимальное количество точек в облаке")
    parser.add_argument("--checkpoint_freq", type=int, default=50,
                        help="Частота сохранения чекпоинтов (в батчах)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Количество процессов для загрузки данных")
    parser.add_argument("--skip_pretrained", action="store_true",
                        help="Пропустить загрузку предварительно обученной модели (только для отладки)")
    return parser.parse_args()

def validate_model(model, val_dataloader, device, diffusion):
    """Валидация модели"""
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                with autocast():
                    if model.model_type == "text300M":
                        model_kwargs = {'texts': batch.get('text', [''] * len(batch['points']))}
                    else:
                        model_kwargs = {'images': batch['image']}
                    
                    x_start = batch['points']
                    t = torch.randint(0, diffusion.num_timesteps, (len(x_start),), device=device)
                    losses = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
                    loss = losses['loss'].mean()
                    
                total_val_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error during validation: {e}")
                continue
    
    return total_val_loss / num_batches if num_batches > 0 else float('inf')

def load_and_customize_config(args):
    """
    Загружает и настраивает конфигурацию модели на основе аргументов командной строки
    и доступных ресурсов
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        dict: Настроенная конфигурация модели
    """
    # Создаем базовую конфигурацию по умолчанию
    config = {
        "n_ctx": 2048,
        "width": 256,
        "layers": 6,
        "heads": 4,
        "dropout": 0.1,
        "text_embed_dim": 256,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "input_channels": 3,
        "output_channels": 3,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_timesteps": 1000,
        "model_mean_type": "epsilon",
        "model_var_type": "fixed_small",
        "loss_type": "mse"
    }
    
    # Читаем конфигурацию из файла, если он существует
    if args.config:
        try:
            logger.info(f"Пробуем загрузить конфигурацию из {args.config}")
            with open(args.config, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    config.update(loaded_config)
                    logger.info(f"Конфигурация успешно загружена из {args.config}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации из {args.config}: {str(e)}")
            logger.warning("Используем базовую конфигурацию")
    
    # Применяем автоматическое масштабирование на основе доступной памяти GPU, если включено
    if args.auto_scale and args.device != 'cpu' and torch.cuda.is_available():
        try:
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Автоматическое масштабирование для GPU с {available_memory_gb:.2f}GB памяти")
            
            # Получаем профиль масштабирования
            scaling_profile = get_memory_scaling_profile(available_memory_gb)
            
            # Применяем профиль к конфигурации
            for key, value in scaling_profile.items():
                if key in config:
                    config[key] = value
                    logger.info(f"Авто-масштабирование: {key} = {value}")
                elif hasattr(args, key):
                    setattr(args, key, value)
                    logger.info(f"Авто-масштабирование аргумента: {key} = {value}")
        except Exception as e:
            logger.warning(f"Ошибка при автоматическом масштабировании: {str(e)}")
    
    # Переопределяем параметры из командной строки (если указаны)
    override_params = [
        ("n_ctx", args.n_ctx),
        ("width", args.width),
        ("layers", args.layers),
        ("heads", args.heads),
        ("dropout", args.dropout),
        ("text_embed_dim", args.text_embed_dim)
    ]
    
    for param_name, param_value in override_params:
        if param_value is not None:
            original_value = config.get(param_name, "не задано")
            config[param_name] = param_value
            logger.info(f"Переопределяем {param_name}: {original_value} -> {param_value}")
    
    # Сохраняем настроенную конфигурацию для воспроизводимости
    os.makedirs(args.output_dir, exist_ok=True)  # Убедимся, что директория существует
    customized_config_path = os.path.join(args.output_dir, f"{args.model_type}_customized.yaml")
    try:
        with open(customized_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        logger.info(f"Настроенная конфигурация сохранена в {customized_config_path}")
    except Exception as e:
        logger.error(f"Не удалось сохранить настроенную конфигурацию: {str(e)}")
    
    return config

def train_model(args):
    """Оптимизированная функция обучения модели для GTX 1660 Super"""
    logger.info(f"Начало обучения модели {args.model_type} на устройстве {args.device}")
    
    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Если указан флаг prepare_data, сначала подготавливаем данные
    if args.prepare_data:
        logger.info(f"Подготовка данных: загрузка {args.num_objects} объектов")
        try:
            prepare_objaverse_data(
                output_dir=args.data_dir,
                num_objects=args.num_objects,
                max_points=args.max_points,
                batch_size=args.batch_size
            )
            logger.info("Подготовка данных завершена")
            return  # Завершаем выполнение после подготовки данных
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            raise
    
    # Настройка CUDA для оптимальной производительности
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # Отключаем TF32, так как GTX 1660 Super его не поддерживает
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Проверяем доступную память GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logger.info(f"Доступно памяти GPU: {gpu_memory:.2f}GB")
        
        if gpu_memory < 6.5:  # Если памяти меньше 6.5GB
            args.batch_size = 1
            args.gradient_accumulation_steps = 8
            logger.warning("Недостаточно памяти GPU, уменьшаем batch_size и увеличиваем gradient_accumulation_steps")
    
    # Загружаем конфигурацию и модель
    try:
        device = torch.device(args.device)
        
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            diffusion = diffusion_from_config(config)
        else:
            # Для отладки пропускаем загрузку конфигурации
            if args.skip_pretrained:
                # Создаем заглушку для диффузии
                betas = torch.linspace(0.0001, 0.02, 1000)  # CPU tensor
                diffusion = GaussianDiffusion(
                    betas=betas.numpy(),  # Преобразуем в numpy array
                    model_mean_type="epsilon",
                    model_var_type="fixed_small",
                    loss_type="mse"
                )
            else:
                diffusion = diffusion_from_config(load_config('diffusion'))
        
        if args.skip_pretrained:
            # Создаем заглушку для модели для отладки
            from shap_e.models.generation.transformer import PointDiffusionTransformer
            from shap_e.models.generation.transformer import CLIPImagePointDiffusionTransformer
            logger.warning("Используется тестовая модель (без предварительного обучения)")
            
            # Выбираем модель в зависимости от типа
            if args.model_type == "text300M":
                # Создаем простую текстовую модель (для отладки)
                class TextConditionedTransformer(PointDiffusionTransformer):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.model_type = "text300M"
                        
                    def forward(self, x, t, texts=None, **kwargs):
                        # Игнорируем текстовое условие в тестовой модели
                        return super().forward(x, t)
                
                model = TextConditionedTransformer(
                    device=device,
                    dtype=torch.float32,
                    input_channels=3,
                    output_channels=3,
                    n_ctx=args.max_points
                )
            else:
                # Создаем простую модель для изображений
                model = PointDiffusionTransformer(
                    device=device,
                    dtype=torch.float32,
                    input_channels=3,
                    output_channels=3,
                    n_ctx=args.max_points
                )
        else:
            model = load_model(args.model_type, device=device)
        
        model.train()
        
    except Exception as e:
        logger.error(f"Error loading model and config: {e}")
        raise
    
    # Загружаем датасет
    try:
        train_dataloader = get_objaverse_dataloader(
            args.data_dir, 
            args.batch_size,
            split='train',
            num_workers=args.num_workers,
            max_points=args.max_points
        )
        
        val_dataloader = get_objaverse_dataloader(
            args.data_dir,
            args.batch_size,
            split='val',
            num_workers=2,  # Меньше workers для валидации
            max_points=args.max_points
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Настраиваем оптимизатор и scaler для mixed precision
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # Добавляем планировщик скорости обучения
    steps_per_epoch = max(1, len(train_dataloader) // args.gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Быстрый разогрев в начале
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Для возобновления обучения
    best_val_loss = float('inf')
    global_step = 0
    
    try:
        for epoch in range(args.epochs):
            logger.info(f"Эпоха {epoch+1}/{args.epochs}")
            model.train()
            
            # Добавляем прогресс-бар с дополнительной информацией
            pbar = tqdm(train_dataloader, desc=f"Эпоха {epoch+1}")
            running_loss = 0.0
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Очистка памяти каждые 10 батчей
                    if batch_idx % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # Перемещаем данные на GPU
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Mixed precision training
                    with autocast():
                        if args.model_type == "text300M":
                            model_kwargs = {'texts': batch.get('text', [''] * len(batch['points']))}
                        else:
                            model_kwargs = {'images': batch['image']}
                        
                        x_start = batch['points']
                        t = torch.randint(0, diffusion.num_timesteps, (len(x_start),), device=device)
                        
                        losses = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
                        loss = losses['loss'].mean() / args.gradient_accumulation_steps
                    
                    # Накопление градиентов
                    scaler.scale(loss).backward()
                    running_loss += loss.item() * args.gradient_accumulation_steps
                    
                    if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                        
                        # Обновляем прогресс-бар
                        pbar.set_postfix({
                            'loss': running_loss / args.gradient_accumulation_steps,
                            'lr': scheduler.get_last_lr()[0]
                        })
                        running_loss = 0.0
                        
                        global_step += 1
                        
                        # Валидация и сохранение
                        if global_step % args.checkpoint_freq == 0:
                            val_loss = validate_model(model, val_dataloader, device, diffusion)
                            logger.info(f"Step {global_step}, Val Loss: {val_loss:.4f}")
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'val_loss': val_loss,
                                    'global_step': global_step,
                                }, os.path.join(args.output_dir, "best_model.pt"))
                            
                            model.train()
                
                except Exception as e:
                    logger.error(f"Error in training step: {e}")
                    continue
            
            # Сохраняем чекпоинт в конце эпохи
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': best_val_loss,
                'global_step': global_step,
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            
            # Очистка памяти в конце эпохи
            gc.collect()
            torch.cuda.empty_cache()
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        # Сохраняем аварийный чекпоинт
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': best_val_loss,
            'global_step': global_step,
        }, os.path.join(args.output_dir, "emergency_checkpoint.pt"))
        raise
    
    finally:
        # Сохраняем финальную модель
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': best_val_loss,
            'global_step': global_step,
        }, os.path.join(args.output_dir, "final_model.pt"))

if __name__ == "__main__":
    args = parse_args()
    train_model(args) 