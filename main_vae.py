from src.model_vae import *
from src.train_vae import *
import random
import json

random.seed(42)
with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)
    random.shuffle(dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(base_channels=64).to(device)
print(f"模型大小: {sum(p.numel() for p in vae.parameters()) * 4 / 1024 / 1024:.2f} MB")


# train
train_vae(vae, dataset, device, batch_size=20, epochs=30, lr=1e-3, save_path="VAE_64.pth")

# inference
#vae.load_state_dict(torch.load("VAE_64.pth"))
#vae_reconstruction(vae, dataset, device, num_images=8)

