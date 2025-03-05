import torch
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional

class TextEncoder:
    """Класс для кодирования текстовых промптов в эмбеддинги."""
    
    def __init__(
        self,
        device: torch.device,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77
    ):
        """
        Инициализация текстового энкодера.
        
        Args:
            device: Устройство для вычислений (CPU/GPU)
            model_name: Название предобученной модели CLIP
            max_length: Максимальная длина токенизированного текста
        """
        self.device = device
        self.max_length = max_length
        
        print(f"Загрузка текстового энкодера {model_name}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        
        # Замораживаем веса для инференса
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def encode(
        self,
        prompts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Кодирование текстовых промптов в эмбеддинги.
        
        Args:
            prompts: Список текстовых промптов
            normalize: Нормализовать ли выходные эмбеддинги
            
        Returns:
            Тензор эмбеддингов размера (batch_size, embedding_dim)
        """
        # Токенизация текста
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Получение эмбеддингов
        with torch.no_grad():
            embeddings = self.text_encoder(**tokens).last_hidden_state
            
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
        return embeddings
        
    def encode_pooled(
        self,
        prompts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Кодирование текстовых промптов в эмбеддинги с пулингом.
        
        Args:
            prompts: Список текстовых промптов
            normalize: Нормализовать ли выходные эмбеддинги
            
        Returns:
            Тензор эмбеддингов размера (batch_size, embedding_dim)
        """
        # Токенизация текста
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Получение эмбеддингов с пулингом
        with torch.no_grad():
            embeddings = self.text_encoder(**tokens).pooler_output
            
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
        return embeddings 