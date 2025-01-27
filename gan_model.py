'''
файл для создания gan модели, состоящий из двух главных частей - генератор и дискриминатор.
более подробно про них почитайте в инете чтобы понимать общий принцип gan
'''

import torch
import torch.nn as nn
from tqdm import tqdm  # Импорт библиотеки для прогресс-баров

class Generator(nn.Module):
    def __init__(self, noise_dim=100, embedding_dim=512):
        super().__init__()
        self.tnet = TNet(in_dim=3)
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim

        # Блок для объединения шума и эмбеддинга
        self.fc = nn.Linear(noise_dim + embedding_dim, 3 * 1024)

        # PointNet-слои
        self.pointnet = nn.Sequential(
            nn.Conv1d(3, 64, 1),  # Обрабатывает каждую точку (x, y, z)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        # Глобальный макс-пулинг (агрегация признаков)
        self.global_pool = nn.MaxPool1d(kernel_size=1024)

        # Финал генератора
        self.final_layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024 * 3),  # 1024 точки × 3 координаты
            nn.Tanh()
        )

    def forward(self, noise, embedding):
        # Объединяем шум и эмбеддинг
        combined = torch.cat([noise, embedding], dim=1)
        x = self.fc(combined)
        
        # Преобразуем в формат (batch, channels, points)
        x = x.view(-1, 3, 1024)  # channels=3 (x, y, z), points=1024

        transform_matrix = self.tnet(x)
        x = torch.bmm(transform_matrix, x)
        
        # Обработка PointNet-слоями
        x = self.pointnet(x)
        
        # Глобальный макс-пулинг
        global_features = self.global_pool(x)  # (batch, 256, 1)
        global_features = global_features.view(-1, 256)
        
        # Генерация точек
        points = self.final_layer(global_features)
        points = points.view(-1, 1024, 3)  # (batch, 1024, 3)
        
        return points

class Discriminator(nn.Module):
    def __init__(self, data_dim=3072, embedding_dim=512):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim

        self.model = nn.Sequential(
            nn.Linear(data_dim + embedding_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, data, embedding):
        combined_input = torch.cat((data.view(data.size(0), -1), embedding), dim=1)
        return self.model(combined_input)

def train_gan(generator, discriminator, dataloader, embedding_generator, epochs, lr, device):
    generator.train()
    discriminator.train()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    # Прогресс-бар для эпох
    epoch_progress = tqdm(range(epochs), desc="Обучение GAN", unit="epoch")

    for epoch in epoch_progress:
        total_loss_D = 0.0
        total_loss_G = 0.0
        batches = 0

        # Прогресс-бар для батчей внутри эпохи
        batch_progress = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{epochs}", leave=False, unit="batch")

        for real_data, _, class_names in batch_progress:
            if real_data.size(0) == 0:
                continue

            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Генерация эмбеддингов
            text = class_names[0]
            embedding = embedding_generator.generate_embedding(text).to(device)
            embedding = embedding.expand(batch_size, -1)

            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            # Обучение дискриминатора
            optimizer_D.zero_grad()
            real_preds = discriminator(real_data, embedding)
            loss_real = criterion(real_preds, real_labels)

            noise = torch.randn(batch_size, generator.noise_dim).to(device)
            fake_data = generator(noise, embedding)
            fake_preds = discriminator(fake_data.detach(), embedding)
            loss_fake = criterion(fake_preds, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # Обучение генератора
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, generator.noise_dim).to(device)
            fake_data = generator(noise, embedding)
            fake_preds = discriminator(fake_data, embedding)
            loss_G = criterion(fake_preds, real_labels)
            loss_G.backward()
            optimizer_G.step()

            # Обновление статистики
            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            batches += 1

            # Обновление прогресс-бара батчей
            batch_progress.set_postfix({
                "Loss D": f"{loss_D.item():.4f}",
                "Loss G": f"{loss_G.item():.4f}"
            })

        # Обновление прогресс-бара эпох
        epoch_progress.set_postfix({
            "Avg Loss D": f"{total_loss_D/batches:.4f}",
            "Avg Loss G": f"{total_loss_G/batches:.4f}"
        })

class TNet(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.in_dim = in_dim
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        self.global_pool = nn.MaxPool1d(kernel_size=1024)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, in_dim * in_dim)  # Матрица in_dim x in_dim
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)  # (batch, 256, 1024)
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.view(batch_size, -1)  # (batch, 256)
        matrix = self.fc_layers(x)  # (batch, in_dim * in_dim)
        matrix = matrix.view(batch_size, self.in_dim, self.in_dim)  # (batch, 3, 3)
        return matrix