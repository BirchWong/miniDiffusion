import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class FlowMatching(nn.Module):
    def __init__(self, model, device='cuda'):
        super(FlowMatching, self).__init__()
        self.model = model
        self.device = device

    def add_noise(self, x_1, t, x_0=None):
        """
        x_0 : noise
        x_1 : image
        x_t = (1-t) * x_0 + t * x_1
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        # t : (batch_size,) -> (batch_size, 1, 1, 1)
        t = t.view(-1, 1, 1, 1)

        x_t = (1 - t) * x_0 + t * x_1

        return x_t, x_0

    def compute_loss(self, x_1, t, label, x_0=None):
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        x_t, x_0 = self.add_noise(x_1, t, x_0)

        target_velocity = x_1 - x_0

        predicted_velocity = self.model(x_t, t, label)

        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample(self, noise=None,label=None, num_steps=100):
        """noise -> image"""
        x = noise
        bs, c, h, w = x.shape
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((bs,), i * dt, device=self.device, dtype=torch.float32)
            velocity = self.model(x, t, label)
            x = x + velocity * dt

        return x

    def forward(self, x, t, label):
        return self.model(x, t, label)

#   这里同样提供了DDPM算法，用法和FlowMatching完全一样。
#   self函数形参名称不一样，但按顺序指代相同内容。不指定形参名，接口完全相同
#   两者U-net预测不一样的东西，所以loss指代着不同的内容。

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super(DDPM, self).__init__()

        self.model = model
        self.timesteps = timesteps
        self.device = device

        # beta_t 这里更准确的说是，所有t的beta。其他定义同理。 [beta_1,beta_2,...,beta_t]
        self.b = torch.linspace(beta_start, beta_end, timesteps)
        self.b = self.b.to(device).view(-1, 1, 1, 1)

        # alpha_t
        self.a = 1.0 - self.b
        # alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
        self.a_bar = torch.cumprod(self.a, dim=0)
        # alpha_bar_t-1
        self.a_bar_prev = torch.cat([torch.ones(1, 1, 1, 1, device=self.a_bar.device), self.a_bar[:-1]], dim=0)

        # beta_tilde_t = beta_t * (1-alpha_bar_t-1) / (1-alpha_bar_t)
        self.b_tilde = self.b * (1.0 - self.a_bar_prev) / (1.0 - self.a_bar)
        # log(b_tilde_t)
        self.log_b_tilde = torch.log(self.b_tilde.clamp(min=1e-20))

        # sqrt(a_bar_t)
        self.sqrt_a_bar = torch.sqrt(self.a_bar)
        # sqrt(1-a_bar_t)
        self.sqrt_one_minus_a_bar = torch.sqrt(1.0 - self.a_bar)

    #   q_sample
    def add_noise(self, x_0, t, noise=None):
        """
        正向：任意步加噪声
        x_0 : image
        x_t = sqrt(a_bar_t) * x_0 + sqrt(1-a_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        return self.sqrt_a_bar[t] * x_0 + self.sqrt_one_minus_a_bar[t] * noise, noise

    #   p_losses
    def compute_loss(self, x_0, t, label, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t, noise = self.add_noise(x_0, t, noise)
        predicted_noise = self.model(x_t, t, label)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t, label):
        """ 反向：单步去除噪声 (x_t -> x_t-1) """
        predicted_noise = self.model(x_t, t, label)

        # mean = sqrt(1/a_t) * ( x_t - (b_t/sqrt(1-a_bar_t)*noise) )
        mean = torch.sqrt(1.0 / self.a[t]) * (x_t - self.b[t]  / self.sqrt_one_minus_a_bar[t] * predicted_noise)

        if t[0] == 0:
            x_prev = mean
        else:
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(self.b_tilde[t]) * noise
        return x_prev

    @torch.no_grad()
    def sample(self, noise, label):
        """ noise->image 逐步去噪 """
        x = noise
        bs, c, h, w = x.shape
        # 逐步去噪
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            t = torch.full((bs,), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t, label)

        return x

    def forward(self, x, t, label):
        return self.model(x, t, label)