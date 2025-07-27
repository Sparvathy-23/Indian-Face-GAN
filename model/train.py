import os
import torch
from parquet_dataset import ParquetImageDataset
from training_loop import train_gan
from generator import Generator
from discriminator import Discriminator
from weights_init import weights_init
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_gan_model(model_path):
    """Load a saved GAN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model info
    model_info = torch.load(model_path, map_location=device)

    # Initialize the generator with the same latent and text embedding dimensions
    generator = Generator(model_info['latent_dim'], model_info['text_embed_dim']).to(device)
    generator.load_state_dict(model_info['generator_state'])
    generator.eval()  # Set to evaluation mode

    return generator, model_info['latent_dim'], model_info['text_embed_dim']

def generate_images_from_prompts(generator, latent_dim, prompts, device=None):
    """Generate images conditioned on prompts using the trained generator"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and embedder
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    embedder = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", output_hidden_states=True)
    embedder.eval()
    embedder.to(device)

    generator.eval()
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(len(prompts), latent_dim, device=device)
        
        # Embed prompts
        text_embeddings = []
        for prompt in prompts:
            tokens = tokenizer(
                prompt,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=100  # Fixed length
            ).to(device)
            
            # Get embeddings using hidden states
            outputs = embedder(**tokens)
            hidden_states = outputs.hidden_states[-1]
            embedding = hidden_states.mean(dim=1).squeeze(0)
            text_embeddings.append(embedding)
            
        text_embeddings = torch.stack(text_embeddings)

        # Generate images
        generated_images = generator(noise, text_embeddings)

        # Convert images from tensor to PIL Image
        generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)
        images = []
        for img_tensor in generated_images.cpu():
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img = Image.fromarray((img_array * 255).astype('uint8'))
            images.append(img)

        return images

def generate_and_save_images_from_prompts(model_path, prompts, output_dir='generated_images'):
    """Load model, generate images from prompts, and save them"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    generator, latent_dim, _ = load_gan_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate images
    images = generate_images_from_prompts(generator, latent_dim, prompts, device=device)

    # Save images
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f'generated_image_{i+1}.png'))

    return images

if __name__ == "__main__":
    # Training parameters
    parquet_path = r"/kaggle/input/indian-forensic/combined_file.parquet"
    save_dir = '/kaggle/working/'
    
    # Initialize transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create dataset
    dataset = ParquetImageDataset(parquet_path, transform=transform)

    # First, train the model
    print("Training GAN...")
    # Check for latest checkpoint
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.startswith('checkpoint_epoch_')])
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoints_dir, checkpoints[-1])
            print(f"Found checkpoint: {latest_checkpoint}")
            generator, discriminator = train_gan(
                dataset=dataset,
                num_epochs=5000,
                batch_size=16,
                latent_dim=100,
                text_embed_dim=768,
                save_dir=save_dir,
                checkpoint_path=latest_checkpoint
            )
        else:
            print("No checkpoint found, starting fresh training")
            generator, discriminator = train_gan(
                dataset=dataset,
                num_epochs=5000,
                batch_size=16,
                latent_dim=100,
                text_embed_dim=768,
                save_dir=save_dir
            )
    else:
        print("No checkpoints directory found, starting fresh training")
        generator, discriminator = train_gan(
            dataset=dataset,
            num_epochs=5000,
            batch_size=16,
            latent_dim=100,
            text_embed_dim=768,
            save_dir=save_dir
        )

    # Then generate some images using the trained model
    print("\nGenerating images from trained model...")
    model_path = os.path.join(save_dir, 'gan_model_final.pth')
    output_dir = 'generated_images'

    # Define prompts for image generation
    prompts = [
        "A adult male, from North India, with a diamond face shape, medium black hair styled as straight, black eyes, with a straight nose, with thin lips, with thick eyebrows, with medium ears.",
        "A adult male, from North India, with a diamond face shape, medium black hair styled as straight, black eyes, with a straight nose, with medium lips, with straight eyebrows, with small ears.",
        "A adult male, from North India, with an oval face shape, short black hair styled as straight, black eyes, with a straight nose, with medium lips, with thick eyebrows, with small ears."
    ]

    # Generate and save images from prompts
    generated_images = generate_and_save_images_from_prompts(model_path, prompts, output_dir=output_dir)
    print(f"Generated images have been saved to {output_dir}/")
