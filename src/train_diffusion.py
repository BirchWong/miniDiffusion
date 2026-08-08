import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from transformers import get_cosine_schedule_with_warmup
import copy


def train_diffusion(diffusion, dataloader, device, epochs, lr, save_path, generator, checkpoint=None):

    epoch_generate_flags = [1, 2, 3, 5, 8, 13, 21, 34]

    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=lr)

    warmup_epochs = int( epochs * 0.08 )  # 8%的目标 epoch 用来warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs * len(dataloader),
        num_training_steps=epochs * len(dataloader)
    )

    # 初始化EMA
    ema_decay = 0.999
    ema_model = copy.deepcopy(diffusion.model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    start_epoch = 0
    if checkpoint is not None:
        diffusion.model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        target_epoch = checkpoint['target_epoch']
        epochs = target_epoch - start_epoch
    else:
        target_epoch = epochs

    if target_epoch == start_epoch: return 0

    logging.basicConfig(level=logging.INFO, format='\033[37m%(message)s\033[0m')
    logger = logging.getLogger(__name__)

    for epoch in range(epochs):
        epoch = epoch + start_epoch
        total_loss = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs+start_epoch}')
        for latent, label in pbar:
            latent = latent.to(device)
            label = label.to(device)

            #print(f"Latent mean: {latent.mean().item():.6f}, std: {latent.std().item():.6f}")

            bs,c,h,w = latent.shape

            if diffusion.__class__.__name__ == "DDPM":
                t = torch.randint(0, diffusion.timesteps, (bs,), device=device)
            elif diffusion.__class__.__name__ == "FlowMatching":
                t = torch.rand(bs, device=device)

            loss = diffusion.compute_loss(latent, t, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # θ_EMA_new = decay × θ_EMA_old + (1 - decay) × θ_model
            with torch.no_grad():
                for ema_param, model_param in zip(ema_model.parameters(), diffusion.model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch + 1}/{epochs+start_epoch}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        if (epoch+1) % 10 == 0:
            checkpoint = {
                "target_epoch": target_epoch,
                "epoch": epoch+1,
                "model": diffusion.model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint, save_path)

        if (epoch + 1) % 50 == 0 or (epoch + 1) in epoch_generate_flags:
            # 推理的时候把ddpm的model 换成 ema，这里是换软链接
            original_model = diffusion.model
            diffusion.model = ema_model
            if diffusion.__class__.__name__ == "DDPM":
                generator.generate(save_dir=r'DDPM_output',filename=f"DDPM_generation_{epoch+1}_{avg_loss:.4f}.png")
            if diffusion.__class__.__name__ == "FlowMatching":
                generator.generate(save_dir=r'FlowMatching_output',filename=f"FlowMatching_generation_{epoch+1}_{avg_loss:.4f}.png")
            diffusion.model = original_model

    checkpoint = {
        "target_epoch": target_epoch,
        "epoch": target_epoch,
        "model": diffusion.model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)


def vae_encode_to_latent(vae, dataset, prompt_embedding_dictionary, device, batch_size, save_path='dataset_for_diffusion.pt'):
    vae.eval()
    latents = []
    labels = []

    # 按照每个batch送入vae处理
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        x = []
        index = []
        for item in batch:
            img = Image.open(item["image_path"])
            img = img.resize((256, 256))
            img_array = np.array(img) / 255.0 * 2 - 1  # [-1,1]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            x.append(img_tensor)
            index.append(item["index"])
        x = torch.stack(x).to(device)
        with torch.no_grad():
            mu, logvar = vae.encode(x)

        latent = mu.cpu()
        label = prompt_embedding_dictionary[index]
        label = torch.from_numpy(label)
        latents.append(latent)
        labels.append(label)

    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)

    torch.save({
        'latents': latents,
        'labels': labels
    }, save_path)