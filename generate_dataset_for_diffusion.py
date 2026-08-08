from src.model_vae import *
from src.train_diffusion import *
import torch
import json

with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

prompt_embedding_dictionary = np.load("prompt_embedding_dictionary_T5.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(base_channels=64).to(device)
vae.load_state_dict(torch.load("vae_64.pth"))
vae_encode_to_latent(vae, dataset, prompt_embedding_dictionary, device, batch_size=128, save_path='dataset_for_diffusion.pt')
