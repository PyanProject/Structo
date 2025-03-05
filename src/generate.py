import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import os
from PIL import Image
import logging
from datetime import datetime
import argparse

from text_encoder import TextEncoder
from diffusion import UNet3DConditioned, DiffusionModel
from decoder_3d import Decoder3D, NeRFDecoder
from renderer import VolumeRenderer
from configs.default import Config

class Generator:
    """Класс для генерации 3D моделей из текстовых описаний."""
    
    def __init__(
        self,
        config: Config,
        device: torch.device,
        checkpoint_path: str
    ):
        """
        Args:
            config: Конфигурация
            device: Устройство для генерации
            checkpoint_path: Путь к чекпоинту модели
        """
        self.config = config
        self.device = device
        
        # Инициализация логгера
        self.setup_logging()
        
        # Инициализация моделей
        self.setup_models()
        
        # Загрузка весов
        self.load_checkpoint(checkpoint_path)
        
    def setup_logging(self):
        """Настройка логирования."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """Инициализация моделей."""
        # Текстовый энкодер
        self.text_encoder = TextEncoder(
            device=self.device,
            model_name=self.config.text_encoder.model_name,
            max_length=self.config.text_encoder.max_length
        )
        
        # U-Net
        self.unet = UNet3DConditioned(
            **vars(self.config.unet)
        ).to(self.device)
        
        # Диффузионная модель
        self.diffusion = DiffusionModel(
            model=self.unet,
            **vars(self.config.diffusion),
            device=self.device
        )
        
        # 3D декодер
        self.decoder_3d = Decoder3D(
            **vars(self.config.decoder_3d)
        ).to(self.device)
        
        # NeRF декодер
        self.nerf = NeRFDecoder(
            **vars(self.config.nerf)
        ).to(self.device)
        
        # Рендерер
        self.renderer = VolumeRenderer(
            **vars(self.config.renderer)
        ).to(self.device)
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Загрузка весов модели.
        
        Args:
            checkpoint_path: Путь к чекпоинту
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint['unet_state'])
        self.decoder_3d.load_state_dict(checkpoint['decoder_3d_state'])
        self.nerf.load_state_dict(checkpoint['nerf_state'])
        
        self.logger.info(f'Загружен чекпоинт: {checkpoint_path}')
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        save_mesh: bool = True,
        save_preview: bool = True
    ) -> Tuple[torch.Tensor, Optional[str], Optional[str]]:
        """
        Генерация 3D модели из текстового описания.
        
        Args:
            prompt: Текстовое описание
            output_dir: Директория для сохранения результатов
            save_mesh: Сохранять ли меш
            save_preview: Сохранять ли превью
            
        Returns:
            latents: Латентное представление
            mesh_path: Путь к сохраненному мешу (если save_mesh=True)
            preview_path: Путь к сохраненному превью (если save_preview=True)
        """
        self.logger.info(f'Генерация модели для промпта: "{prompt}"')
        
        # Создаем директорию для результатов
        if output_dir is None:
            output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Получаем текстовые эмбеддинги
        text_embeddings = self.text_encoder.encode([prompt])
        
        # Генерируем латентное представление
        latents = self.diffusion.sample(
            batch_size=1,
            condition=text_embeddings,
            cfg_scale=self.config.generation.cfg_scale
        )
        
        mesh_path = None
        if save_mesh:
            # Декодируем в 3D представление
            occupancy, features = self.decoder_3d(latents)
            
            # Извлекаем меш
            mesh = self.decoder_3d.extract_mesh(
                occupancy[0],
                features[0] if features is not None else None
            )
            
            # Сохраняем меш
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mesh_path = os.path.join(output_dir, f'mesh_{timestamp}.obj')
            self.decoder_3d.save_mesh(mesh, mesh_path)
            self.logger.info(f'Меш сохранен: {mesh_path}')
            
        preview_path = None
        if save_preview:
            # Рендерим превью
            colors, depths = self.renderer.render_views(
                self.nerf,
                latents,
                n_views=self.config.generation.num_views
            )
            
            # Сохраняем превью
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_path = os.path.join(output_dir, f'preview_{timestamp}.gif')
            
            # Создаем GIF
            frames = []
            for i in range(colors.shape[1]):
                img = colors[0, i].cpu().numpy()
                img = (img * 255).astype('uint8')
                frames.append(Image.fromarray(img))
                
            # Сохраняем GIF
            frames[0].save(
                preview_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
            self.logger.info(f'Превью сохранено: {preview_path}')
            
        return latents, mesh_path, preview_path
        
def main():
    """Основная функция для запуска генерации."""
    parser = argparse.ArgumentParser(description='Генерация 3D моделей из текстовых описаний')
    
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Текстовое описание для генерации'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Путь к чекпоинту модели'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Директория для сохранения результатов'
    )
    
    parser.add_argument(
        '--no_mesh',
        action='store_true',
        help='Не сохранять меш'
    )
    
    parser.add_argument(
        '--no_preview',
        action='store_true',
        help='Не сохранять превью'
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = Config()
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем генератор
    generator = Generator(config, device, args.checkpoint)
    
    # Генерируем модель
    generator.generate(
        prompt=args.prompt,
        output_dir=args.output_dir,
        save_mesh=not args.no_mesh,
        save_preview=not args.no_preview
    )
    
if __name__ == '__main__':
    main() 