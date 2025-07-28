import torch
from model.generator import Generator
from model.discriminator import Discriminator
from model.text_encoder import TextEncoder
from model.train import train_gan
from utils.data_loader import get_dataloader

def main():
    # Hyperparameters
    latent_dim = 100
    embed_dim = 768  # BERT embedding size
    image_size = 64
    batch_size = 32
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    dataloader = get_dataloader(batch_size=batch_size)

    print("Initializing models...")
    generator = Generator(latent_dim, embed_dim).to(device)
    discriminator = Discriminator(embed_dim).to(device)
    text_encoder = TextEncoder().to(device)

    print("Starting training...")
    train_gan(generator, discriminator, text_encoder, dataloader, latent_dim, epochs, device)

if _name_ == "_main_":
    main()
