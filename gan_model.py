import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, embedding_dim=512, output_dim=3072):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim

        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, embedding):
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        combined_input = torch.cat((noise, embedding), dim=1)
        x = self.model(combined_input)
        return x.view(-1, 3072)


class Discriminator(nn.Module):
    def __init__(self, data_dim, embedding_dim=512):
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

    for epoch in range(epochs):
        total_loss_D = 0.0
        total_loss_G = 0.0
        batches = 0

        for real_data in dataloader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Generate random text embeddings
            text = "sample text"  # Replace with actual text generation
            embedding = embedding_generator.generate_embedding(text).to(device)
            embedding = embedding.squeeze()
            embedding = embedding.expand(batch_size, -1)  # Expand embedding to match batch size

            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            optimizer_D.zero_grad()

            # Real data with real embeddings
            real_preds = discriminator(real_data, embedding)
            loss_real = criterion(real_preds, real_labels)

            # Fake data with real embeddings
            noise = torch.randn(batch_size, generator.noise_dim).to(device)
            fake_data = generator(noise, embedding)
            fake_preds = discriminator(fake_data.detach(), embedding)
            loss_fake = criterion(fake_preds, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Generate fake data with real embeddings
            noise = torch.randn(batch_size, generator.noise_dim).to(device)
            fake_data = generator(noise, embedding)
            fake_preds = discriminator(fake_data, embedding)
            loss_G = criterion(fake_preds, real_labels)
            loss_G.backward()
            optimizer_G.step()

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            batches += 1

        avg_loss_D = total_loss_D / batches
        avg_loss_G = total_loss_G / batches
        print(f"[GAN TRAIN] Эпоха [{epoch + 1}/{epochs}] - Потери D: {avg_loss_D:.4f}, Потери G: {avg_loss_G:.4f}")