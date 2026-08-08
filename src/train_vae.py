import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from .visualize import *
from PIL import Image
from tqdm import tqdm
import lpips
import logging

def vae_loss(recon_x, x, mu, logvar, perceptual_loss_fn):
    #bce_loss = F.binary_cross_entropy(recon_x, x, reduction='mean') # BCE
    #mse_loss = F.mse_loss(recon_x, x, reduction='mean')  # MSE

    mae_loss = F.l1_loss(recon_x, x, reduction='mean')   # L1 | MAE

    lpips_loss = perceptual_loss_fn(recon_x, x).mean()

    recon_loss = mae_loss + 1.0 * lpips_loss

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    '''下面这个比例非常重要'''
    loss = recon_loss + 1e-4 * kl_loss

    return loss, mae_loss, lpips_loss, kl_loss


def train_vae(model, dataset, device, batch_size, epochs, lr, save_path):
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    for param in perceptual_loss_fn.parameters():
        param.requires_grad = False
    perceptual_loss_fn.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    warmup_epochs = 3  # warmup轮数
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs * (len(dataset) // batch_size),
        num_training_steps=epochs * (len(dataset) // batch_size)
    )

    logging.basicConfig(level=logging.INFO, format='\033[37m%(message)s\033[0m')
    logger = logging.getLogger(__name__)

    losses = []
    mae_losses = []
    lpips_losses = []
    kl_losses = []

    for epoch in range(epochs):
        total_loss = 0
        total_mae_loss = 0
        total_lpips_loss = 0
        total_kl_loss = 0

        pbar = tqdm(range(0, len(dataset), batch_size))

        for i in pbar:
            batch = dataset[i:i + batch_size]
            x = []
            for item in batch:
                img = Image.open(item["image_path"])
                img = img.resize((256, 256))
                img_array = np.array(img) / 255.0 * 2 - 1       # [-1,1]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

                x.append(img_tensor)

            x = torch.stack(x)
            x = x.to(device)

            recon_x, mu, logvar = model(x)
            loss, mae_loss, lpips_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, perceptual_loss_fn)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_mae_loss += mae_loss.item()
            total_lpips_loss += lpips_loss.item()
            total_kl_loss += kl_loss.item()

            pbar.set_postfix({
                'avg_loss': f'{loss:.4f}',
            })

        avg_loss = total_loss * batch_size / len(dataset)
        avg_mae_loss = total_mae_loss  * batch_size / len(dataset)
        avg_lpips_loss = total_lpips_loss * batch_size / len(dataset)
        avg_kl_loss = total_kl_loss  * batch_size / len(dataset)

        losses.append(avg_loss)
        mae_losses.append(avg_mae_loss)
        lpips_losses.append(avg_lpips_loss)
        kl_losses.append(avg_kl_loss)

        logger.info(f'Epoch {epoch + 1}: Loss: {avg_loss:.4f}, MAE: {avg_mae_loss:.4f}, '
              f'LPIPS: {avg_lpips_loss:.6f}, KL: {avg_kl_loss:.4f}')

        if (epoch + 1) % 5 == 0:
            filename = f'VAE_reconstruction_{epoch + 1}.png'
            vae_reconstruction(model, dataset, device, num_images=8, filename=filename)

    torch.save(model.state_dict(), save_path)
    print("VAE model saved")
    vae_three_loss_plot(mae_losses, lpips_losses, kl_losses, 'MAE', 'LPIPS', 'KL', 'VAE_training_loss.png')
