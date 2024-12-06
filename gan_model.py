import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_gan(generator, discriminator, dataloader, epochs, lr, device):
    """
    Тренировка GAN с использованием заданного датасета.
    """
    generator.train()
    discriminator.train()

    # Оптимизаторы
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        for real_data, _ in dataloader:  # Путь к файлу игнорируем
            real_data = real_data.to(device)  # Преобразуем эмбеддинги в тензор и переносим на устройство

            batch_size = real_data.size(0)

            # Генерация меток
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            # === Обучение дискриминатора ===
            optimizer_D.zero_grad()

            # Потери для реальных данных
            real_preds = discriminator(real_data)
            loss_real = criterion(real_preds, real_labels)

            # Генерация фейковых данных
            noise = torch.randn(batch_size, generator.input_dim).to(device)
            fake_data = generator(noise)

            # Потери для фейковых данных
            fake_preds = discriminator(fake_data.detach())
            loss_fake = criterion(fake_preds, fake_labels)

            # Итоговые потери дискриминатора
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # === Обучение генератора ===
            optimizer_G.zero_grad()

            # Потери генератора
            fake_preds = discriminator(fake_data)
            loss_G = criterion(fake_preds, real_labels)  # Хотим, чтобы фейковые данные были приняты как реальные
            loss_G.backward()
            optimizer_G.step()

        print(f"Эпоха [{epoch + 1}/{epochs}] - Потери D: {loss_D.item():.4f}, Потери G: {loss_G.item():.4f}")
