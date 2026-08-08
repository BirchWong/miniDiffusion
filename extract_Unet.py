import torch

checkpoint = torch.load('FlowMatching_256.pth')
torch.save(checkpoint['ema_model'], 'FlowMatching_Unet_256.pth')