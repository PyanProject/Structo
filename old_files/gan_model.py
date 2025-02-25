'''
файл для создания gan модели, состоящий из двух главных частей - генератор и дискриминатор.
более подробно про них почитайте в инете чтобы понимать общий принцип gan
'''

import torch
import torch.nn as nn
from tqdm import tqdm  # Импорт библиотеки для прогресс-баров

def chamfer_distance(pc1, pc2):
    """
    Вычисляет Chamfer distance между двумя батчами облаков точек.
    pc1, pc2: тензоры размера (B, N, 3)
    """
    # Вычисляем попарные разности: (B, N, N, 3)
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)
    # Евклидовы расстояния
    dist = torch.norm(diff, dim=-1)  # (B, N, N)
    min1, _ = torch.min(dist, dim=2)  # (B, N)
    min2, _ = torch.min(dist, dim=1)  # (B, N)
    loss = torch.mean(min1) + torch.mean(min2)
    return loss

class Generator(nn.Module):
    def __init__(self, noise_dim=100, embedding_dim=512):
        super().__init__()
        self.tnet = TNet(in_dim=3)
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim

        # Block for combining noise and embedding
        self.fc = nn.Linear(noise_dim + embedding_dim, 3 * 4096)  # Generate 4096 points

        # PointNet layers
        self.pointnet = nn.Sequential(
            nn.Conv1d(3, 64, 1),  # Process each point (x, y, z)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        # Global max pooling (feature aggregation)
        self.global_pool = nn.MaxPool1d(kernel_size=4096)

        # Final layers of generator
        self.final_layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 4096 * 3),  # 4096 points × 3 coordinates
            nn.Tanh()
        )

    def forward(self, noise, embedding):
        # Объединяем шум и эмбеддинг
        combined = torch.cat([noise, embedding], dim=1)
        x = self.fc(combined)
        
        # Reshape to (batch, channels, points)
        x = x.view(-1, 3, 4096)  # channels=3 (x, y, z), points=4096

        transform_matrix = self.tnet(x)
        x = torch.bmm(transform_matrix, x)
        
        # Обработка PointNet-слоями
        x = self.pointnet(x)
        
        # Глобальный макс-пулинг
        global_features = self.global_pool(x)  # (batch, 256, 1)
        global_features = global_features.view(-1, 256)
        
        # Генерация точек
        points = self.final_layer(global_features)
        points = 5.0 * points    # Scale output to enlarge the point cloud
        points = points.view(-1, 4096, 3)  # (batch, 4096, 3)
        
        return points

class Discriminator(nn.Module):
    def __init__(self, data_dim=12288, embedding_dim=512):
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

from sklearn.metrics import pairwise_distances
import numpy as np

def calculate_fid(real_embeddings, fake_embeddings):
    # Убедитесь, что входные данные имеют размерность (n_samples, n_features)
    real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)
    fake_embeddings = fake_embeddings.reshape(fake_embeddings.shape[0], -1)

    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)
    mu_fake = np.mean(fake_embeddings, axis=0)
    sigma_fake = np.cov(fake_embeddings, rowvar=False)

    diff = mu_real - mu_fake
    covmean = np.sqrt(sigma_real.dot(sigma_fake))

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def calculate_is(fake_embeddings):
    # Убедитесь, что входные данные имеют размерность (n_samples, n_features)
    fake_embeddings = fake_embeddings.reshape(fake_embeddings.shape[0], -1)

    # Пример вычисления Inception Score
    kl_divergence = fake_embeddings * (np.log(fake_embeddings) - np.log(np.mean(fake_embeddings, axis=0)))
    is_score = np.exp(np.mean(np.sum(kl_divergence, axis=1)))
    return is_score

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

        real_embeddings = []
        fake_embeddings = []

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
            adv_loss = criterion(fake_preds, real_labels)
            c_loss = chamfer_distance(fake_data, real_data)
            loss_G = adv_loss + 0.1 * c_loss
            loss_G.backward()
            optimizer_G.step()

            # Обновление статистики
            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            batches += 1

            # Сохранение эмбеддингов для метрик
            real_embeddings.append(real_data.cpu().detach().numpy())
            fake_embeddings.append(fake_data.cpu().detach().numpy())

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

        # Вычисление метрик
        real_embeddings = np.concatenate(real_embeddings, axis=0)
        fake_embeddings = np.concatenate(fake_embeddings, axis=0)
        fid = calculate_fid(real_embeddings, fake_embeddings)
        is_score = calculate_is(fake_embeddings)

        print(f"[Epoch {epoch+1}] FID: {fid:.4f}, IS: {is_score:.4f}")

    print('Функция отработала успешно')

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
        self.global_pool = nn.MaxPool1d(kernel_size=4096)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, in_dim * in_dim)  # Матрица in_dim x in_dim
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)  # (batch, 256, 4096)
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.view(batch_size, -1)  # (batch, 256)
        matrix = self.fc_layers(x)  # (batch, in_dim * in_dim)
        matrix = matrix.view(batch_size, self.in_dim, self.in_dim)  # (batch, 3, 3)
        return matrix