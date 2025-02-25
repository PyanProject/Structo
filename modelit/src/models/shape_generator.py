import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelTransformer(nn.Module):
    """
    Трансформер для генерации воксельных представлений 3D объектов
    на основе текстовых эмбеддингов.
    """
    
    def __init__(self, latent_dim=512, hidden_dims=[1024, 512, 256, 128, 64],
                 dropout=0.1, num_heads=8, num_layers=6, voxel_dim=64):
        """
        Инициализация генератора 3D форм.
        
        Args:
            latent_dim (int): Размерность входного латентного пространства.
            hidden_dims (List[int]): Список размерностей скрытых слоев.
            dropout (float): Вероятность дропаута.
            num_heads (int): Количество голов в multi-head attention.
            num_layers (int): Количество слоев трансформера.
            voxel_dim (int): Размерность выходного воксельного представления.
        """
        super(VoxelTransformer, self).__init__()
        
        self.latent_dim = latent_dim
        self.voxel_dim = voxel_dim
        
        # Трансформер-декодер для обработки текстовых эмбеддингов
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dims[0],
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # Преобразование текстового эмбеддинга в начальное скрытое состояние
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Создание тензора позиционных эмбеддингов
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, voxel_dim**3 // 64, latent_dim)
        )
        
        # 3D CNN декодер для генерации воксельной сетки
        self.decoder_cnn = nn.ModuleList()
        
        # Начальная проекция из латентного представления в 4D тензор (B, C, D, H, W)
        self.initial_projection = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4 * 4)
        
        # Создание слоев 3D CNN декодера
        in_channels = hidden_dims[0]
        for i, h_dim in enumerate(hidden_dims[1:]):
            self.decoder_cnn.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = h_dim
        
        # Финальный слой для получения воксельной сетки
        self.final_layer = nn.Conv3d(hidden_dims[-1], 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, text_embeddings):
        """
        Генерирует 3D воксельное представление из текстового эмбеддинга.
        
        Args:
            text_embeddings (torch.Tensor): Тензор текстовых эмбеддингов размера (batch_size, latent_dim).
            
        Returns:
            torch.Tensor: Воксельное представление размера (batch_size, 1, voxel_dim, voxel_dim, voxel_dim).
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        # Проекция текстового эмбеддинга
        latent = self.latent_projection(text_embeddings)
        
        # Расширение размерности для формирования последовательности
        latent = latent.unsqueeze(1)
        
        # Создание позиционных эмбеддингов нужной размерности
        pos_embeddings = self.positional_embedding.expand(batch_size, -1, -1)
        
        # Подготовка памяти для трансформера
        memory = latent.repeat(1, pos_embeddings.shape[1], 1)
        
        # Transformer Decoder
        transformer_output = self.transformer_decoder(pos_embeddings, memory)
        
        # Среднее по последовательности для получения глобального представления
        transformer_output = transformer_output.mean(dim=1)
        
        # Проекция и преобразование размерности
        x = self.initial_projection(transformer_output)
        x = x.view(batch_size, -1, 4, 4, 4)
        
        # Применение 3D CNN декодера
        for layer in self.decoder_cnn:
            x = layer(x)
        
        # Финальный слой и сигмоида для получения вероятностей воксельной сетки
        voxel_grid = torch.sigmoid(self.final_layer(x))
        
        return voxel_grid 