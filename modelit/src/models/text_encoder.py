import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder(nn.Module):
    """
    Модуль для кодирования текстовых промптов с использованием предобученной модели CLIP.
    """
    
    def __init__(self, pretrained=True, freeze=False, embedding_dim=512):
        """
        Инициализация текстового энкодера.
        
        Args:
            pretrained (bool): Использовать предобученные веса.
            freeze (bool): Заморозить параметры модели.
            embedding_dim (int): Размерность выходного эмбеддинга.
        """
        super(TextEncoder, self).__init__()
        
        # Загрузка моделей CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Размерность выходного эмбеддинга CLIP
        clip_embedding_dim = self.text_model.config.hidden_size
        
        # Проекция в нужную размерность
        self.projection = nn.Linear(clip_embedding_dim, embedding_dim)
        
        # Заморозим параметры, если требуется
        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
    
    def forward(self, text_prompts):
        """
        Пропускает текстовые промпты через модель.
        
        Args:
            text_prompts (List[str]): Список текстовых промптов.
            
        Returns:
            torch.Tensor: Тензор с текстовыми эмбеддингами размера (batch_size, embedding_dim).
        """
        # Токенизировать текст
        inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Перенос на нужное устройство
        inputs = {k: v.to(self.text_model.device) for k, v in inputs.items()}
        
        # Получаем выход модели CLIP
        outputs = self.text_model(**inputs)
        
        # Используем [CLS] токен в качестве эмбеддинга
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Проекция в нужную размерность
        text_embeddings = self.projection(text_embeddings)
        
        return text_embeddings 