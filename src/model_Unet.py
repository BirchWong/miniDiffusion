import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    """扩散模型中的时间步编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """残差块"""
    """不改变H,W 只改变C"""
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.net_1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU()
        )
        self.net_2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU()
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        self.time_mlp = nn.Linear(time_dim, out_dim)
        self.silu = nn.SiLU()

    def forward(self, x, t):
        shortcut = self.shortcut(x)
        x = self.net_1(x)
        time_emb = self.silu(self.time_mlp(t))[:, :, None, None]
        x = x + time_emb
        x = self.net_2(x)
        return x + shortcut

class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.res_1 = ResidualBlock(in_dim, out_dim, time_dim)
        self.res_2 = ResidualBlock(out_dim, out_dim, time_dim)
        self.conv = nn.Conv2d(out_dim, out_dim, 4, stride=2, padding=1)
    def forward(self, x, t):
        x = self.res_1(x, t)
        x = self.res_2(x, t)
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_1 = ResidualBlock(in_dim + out_dim, out_dim, time_dim)
        self.res_2 = ResidualBlock(out_dim, out_dim, time_dim)
    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_1(x, t)
        x = self.res_2(x, t)
        return x

class SelfAttention(nn.Module):
    def __init__(self, D, h=32, width=None):
        '''
        H   head_dim
        h   num_heads
        T   seq_len
        c   dim
        '''
        super().__init__()
        self.D =  D # 总维度，即通道
        self.h = h # 头数
        self.H = self.D // self.h # 单头维度
        self.key = nn.Linear(D, D, bias=False)
        self.query = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.proj = nn.Linear(self.H * self.h, self.D)

        self.pos_embed = nn.Parameter(torch.zeros(1,width*width,D))

        self.norm = nn.LayerNorm(D)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x
        B, T, D = x.shape
        x = self.norm(x)
        x = x + self.pos_embed

        k = self.key(x).view(B,T,self.h,self.H).transpose(1,2)    # (B,h,T,H)
        q = self.query(x).view(B,T,self.h,self.H).transpose(1,2)
        v = self.value(x).view(B,T,self.h,self.H).transpose(1,2)

        attn_scores = q @ k.transpose(-2,-1) * self.H **-0.5  # (bs,h,T_Q,H) @ (bs,h,H,T_K) -> (bs,h,T_Q,T_K)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        attn_scores = self.dropout(attn_scores)
        out = attn_scores @ v   # [B, h, T, H]

        out = out.transpose(1, 2).reshape(B, T, D)  # (B,h,T,H) -> (B,T,h,H) -> (B,T,h*H)
        out = self.proj(out)
        return self.dropout(out) + identity   # (B,T,D)


class CrossAttention(nn.Module):
    '''
    text    K V
    image   Q
    B   batch size
    T   sequence length
    D   token dim
    H   head dim
    h   num heads
    '''
    def __init__(self, D, h=32, width=None):
        super().__init__()
        self.h = h
        self.D = D
        self.H = D // h
        self.key = nn.Linear(D, D, bias=False)
        self.query = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.proj = nn.Linear(self.H * self.h, self.D)

        self.pos_embed = nn.Parameter(torch.zeros(1,width*width,D))

        self.norm_q = nn.LayerNorm(D)
        self.norm_k = nn.LayerNorm(D)

        self.dropout = nn.Dropout(0.2)

    # image: x_q
    # text:  x_k     text的位置编码单独进行, T5本身就负责了这部分内容
    def forward(self, x_q,x_k):
        identity = x_q

        x_k = self.norm_k(x_k)
        x_q = self.norm_q(x_q)

        B, T_K, D = x_k.shape
        B, T_Q, D = x_q.shape
        x_q = x_q + self.pos_embed

        k = self.key(x_k).view(B,T_K,self.h,self.H).transpose(1,2)    # (B,h,T,H)
        q = self.query(x_q).view(B,T_Q,self.h,self.H).transpose(1,2)
        v = self.value(x_k).view(B,T_K,self.h,self.H).transpose(1,2)

        attn_scores = q @ k.transpose(-2, -1) * self.H ** -0.5
        attn_scores = torch.softmax(attn_scores, dim=-1)

        attn_scores = self.dropout(attn_scores)
        out = attn_scores @ v   # [B, h, T_Q, H]

        out = out.transpose(1, 2).reshape(B, T_Q, D)  # (B,h,T,H) -> (B,T,h,H) -> (B,T,h*H)
        out = self.proj(out)
        return self.dropout(out) + identity   # (B,T,D)

class FeedFoward(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D),
            nn.Dropout(0.2),
        )
        self.norm = nn.LayerNorm(D)
    def forward(self, x):
        return self.net(self.norm(x)) + x


class Unet(nn.Module):

    """
    [B,   4, 32, 32]    input
    
    [B, 1*b, 32, 32]    conv_in     x1
    [B, 2*b, 16, 16]    down_1      x2
    [B, 4*b,  8,  8]    down_2      x3
    [B, 8*b,  4,  4]    down_3
    
    [B, 8*b,  4,  4]    mid
    res_block + self_attn + cross_attn
    
    [B, 4*b,  8,  8]    up_3
    [B, 2*b, 16, 16]    up_2
    [B, 1*b, 32, 32]    up_1
    [B,   4, 32, 32]    conv_out
    """


    def __init__(self, in_channels=4, out_channels=4, base_channels=64, time_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            PositionEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 下采样
        self.down1 = DownBlock(base_channels * 1, base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_dim)


        # 上采样
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, time_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels * 1, time_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

        # 中间层
        self.text_projector = nn.Linear(768, base_channels * 8) # 768 是 T5-base模型默认
        self.self_attn = SelfAttention(D=base_channels * 8,width=4)
        self.cross_attn = CrossAttention(D=base_channels * 8,width=4)
        self.ffn = FeedFoward(D=base_channels * 8)

    def forward(self, x, t, label):
        t_emb = self.time_mlp(t)

        x1 = self.conv_in(x)

        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x  = self.down3(x3, t_emb)

        # [B,D,width,width] -> [B,D,T] -> [B,T,D]
        B,D,W,W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # transformer block
        x = self.self_attn(x)
        label = self.text_projector(label)   # (B, T=20, 768) -> (B,T_K,base_channels * 8)
        x = self.cross_attn(x,label)
        x = self.ffn(x)

        x = x.transpose(1, 2).reshape(B,D,W,W)

        x = self.up3(x, x3, t_emb)
        x = self.up2(x, x2, t_emb)
        x = self.up1(x, x1, t_emb)

        x = self.conv_out(x)

        return x

