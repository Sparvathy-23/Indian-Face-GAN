class Discriminator(nn.Module):
    def __init__(self, text_embed_dim):
        super(Discriminator, self).__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, True)
        )

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, image, text_embedding):
        img_features = self.main(image).view(-1, 512 * 4 * 4)
        text_features = self.text_proj(text_embedding)
        combined = torch.cat((img_features, text_features), dim=1)
        return torch.sigmoid(self.final(combined)).squeeze()
