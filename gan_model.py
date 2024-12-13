import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, dataloader, epochs, lr, device):
    generator.train()
    discriminator.train()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        total_loss_D = 0.0
        total_loss_G = 0.0
        batches = 0

        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            optimizer_D.zero_grad()
            real_preds = discriminator(real_data)
            loss_real = criterion(real_preds, real_labels)

            noise = torch.randn(batch_size, generator.input_dim).to(device)
            fake_data = generator(noise)
            fake_preds = discriminator(fake_data.detach())
            loss_fake = criterion(fake_preds, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_preds = discriminator(fake_data)
            loss_G = criterion(fake_preds, real_labels)
            loss_G.backward()
            optimizer_G.step()

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            batches += 1

        avg_loss_D = total_loss_D / batches
        avg_loss_G = total_loss_G / batches
        print(f"[GAN TRAIN] Эпоха [{epoch + 1}/{epochs}] - Потери D: {avg_loss_D:.4f}, Потери G: {avg_loss_G:.4f}")
