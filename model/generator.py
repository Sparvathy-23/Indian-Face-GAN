import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, text_embed_dim):
        super(Generator, self).__init__()
        self.noise_proj = nn.Linear(latent_dim, 256)
        self.text_proj = nn.Linear(text_embed_dim, 256)
        self.combined_proj = nn.Linear(512, 8 * 8 * 512)

        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        noise_feat = self.noise_proj(noise)
        text_feat = self.text_proj(text_embedding)
        combined = torch.cat((noise_feat, text_feat), dim=1)
        x = F.leaky_relu(self.combined_proj(combined), 0.2)
        x = x.view(-1, 512, 8, 8)
        return self.main(x)
