import torch
from torch import nn, einsum
import numpy as np
from torch.nn.modules import batchnorm
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        print(x.shape)
        print(self.fn(x, **kwargs).shape)
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.net(x)
        return x.permute(0,3,1,2)


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    distances[:,:,0] += window_size-1
    distances[:,:,1] += window_size-1
    distances[:,:,0] *= 2*window_size-1
    distances_ = distances.sum(-1)
    return distances_


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shuffle, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shuffle = shuffle

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            # self.relative_indices = get_relative_distances(window_size) + window_size - 1
            relative_indices = get_relative_distances(window_size)
            self.register_buffer("relative_indices", relative_indices)
            self.pos_embedding = nn.Parameter(torch.randn((2 * window_size - 1)*(2 * window_size - 1), heads))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        
        if self.shuffle:
            q, k, v = map(
                lambda t: rearrange(t, 'b (w_h nw_h) (w_w nw_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        else:
            q, k, v = map(
                lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale   # 窗口内attention

        if self.relative_pos_embedding:
            # dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
            pos_bias = self.pos_embedding[self.relative_indices.view(-1)].view(self.window_size*self.window_size, self.window_size*self.window_size, -1)
            dots += pos_bias.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(2)
        else:
            dots += self.pos_embedding


        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        if self.shuffle:
            out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (w_h nw_h) (w_w nw_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        else:
            out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        out = out.permute(0,3,1,2)

        return out


class ShuffleBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shuffle, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shuffle=shuffle,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))
        self.local = Residual(PreNorm(dim, nn.Conv2d(dim, dim, window_size, 1, window_size//2, groups=dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.local(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x.permute(0, 3, 1, 2)

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #input of this layernorm is (B,C,H,W), so dim=1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, hidden_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.hidden_dim = hidden_dim
        self.proj = nn.Conv2d(in_chans, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shuffle block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                ShuffleBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shuffle=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                ShuffleBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shuffle=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shuffle_block in self.layers:
            x = regular_block(x)
            x = shuffle_block(x)
        return x


class ShuffleTransformer(nn.Module):
    def __init__(self, *, patch_size, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()
        self.patchembed = PatchEmbed(patch_size=patch_size, in_chans=channels, hidden_dim=hidden_dim)

        self.stage1 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.patchembed(img)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        return self.mlp_head(torch.flatten(x, 1))


def shuffle_t(patch_size=2, hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return ShuffleTransformer(patch_size=patch_size,hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def shuffle_s(patch_size=2, hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return ShuffleTransformer(patch_size=patch_size,hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def shuffle_b(patch_size=2, hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return ShuffleTransformer(patch_size=patch_size, hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def shuffle_l(patch_size=2, hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return ShuffleTransformer(patch_size=patch_size, hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


if __name__ == '__main__':
    net = ShuffleTransformer(
        patch_size=2,
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=10,
        head_dim=32,
        window_size=2,
        downscaling_factors=(2, 2, 2, 2),
        relative_pos_embedding=True
    )
    dummy_x = torch.randn(1, 3, 32, 32)
    logits = net(dummy_x)
    print(logits.size())