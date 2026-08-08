import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def vae_reconstruction(model, dataset, device, num_images=8, filename='VAE_reconstruction.png'):
    model.eval()
    start_index = 0
    batch = dataset[start_index:start_index + num_images]
    x = []
    label = []
    for item in batch:
        img = Image.open(item["image_path"])
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0 * 2 - 1  # [-1,1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        x.append(img_tensor)
        label.append(item["prompt"])
    x = torch.stack(x).to(device)

    with torch.no_grad():
        recon, _, _ = model(x)

    x = x.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
    plt.subplots_adjust(hspace=0.3)

    for i in range(num_images):
        # 原始图像
        img_orig = x[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow((img_orig + 1) / 2)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=24, pad=10)

        # 重建图像
        img_recon = recon[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow((img_recon + 1) / 2)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=24, pad=10)

    plt.tight_layout()
    save_dir = r'VAE_output'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def vae_three_loss_plot(loss_1, loss_2, loss_3, name_1, name_2, name_3, filename='VAE_training_loss.png'):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(loss_1)
    plt.title(name_1)
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 2)
    plt.plot(loss_2)
    plt.title(name_2)
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 3)
    plt.plot(loss_3)
    plt.title(name_3)
    plt.xlabel('Epoch')

    plt.tight_layout()

    save_dir = r'VAE_output'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()


class DiffusionGenerator:
    def __init__(self, vae, diffusion, device, prompts_embedding):
        self.vae = vae
        self.diffusion = diffusion
        self.device = device
        self.prompts_embedding = prompts_embedding

    def generate(self, save_dir=None, filename='Diffusion_generation.png'):
        label = self.prompts_embedding.to(self.device)
        noise = torch.randn(16, 4, 32, 32).to(self.device)
        with torch.no_grad():
            z = self.diffusion.sample(noise=noise, label=label)
            samples = self.vae.decode(z).cpu().detach()

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < 16:
                img = samples[i].permute(1, 2, 0).numpy()
                ax.imshow((img + 1) / 2)
                ax.axis('off')

        plt.suptitle(filename)
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()