# References: https://github.com/alxndrTL/mamba.py/blob/main/mambapy/vim.py

from dataclasses import dataclass
from functools import partial
from typing import Union
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as cp

from .ops.pscan import pscan

#----------------------------------------------------------------------------

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = True # use official CUDA implementation when training (not compatible with (b)float16)

    bidirectional: bool = True # use bidirectional MambaBlock

    divide_output: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

#----------------------------------------------------------------------------

@torch.no_grad()
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

#----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, bias=True):
        super().__init__()
        assert isinstance(img_size, tuple) and len(img_size) == 2
        assert isinstance(patch_size, tuple) and len(patch_size) == 2
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=bias)

    def forward(self, x):
        # B x 3 x H x W -> B x (H' x W') x C
        B, C, H, W = x.shape
        assert H == self.img_size[0]
        assert W == self.img_size[1]
        x = self.proj(x)
        x = torch.flatten(x, start_dim=2).transpose(1, 2)
        return x

#----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

#----------------------------------------------------------------------------

class ViMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # Backward Parameters
        if config.bidirectional:
            A_b = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log_b = nn.Parameter(torch.log(A_b))
            self.A_log_b._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                    kernel_size=config.d_conv, bias=config.conv_bias,
                                    groups=config.d_inner,
                                    padding=config.d_conv - 1)
            
            self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

            self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)

            self.D_b = nn.Parameter(torch.ones(config.d_inner))
            self.D_b._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def inner_forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)
        
        _, L, _ = x.shape
        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)
        x = F.silu(x)
        y = self.ssm(x=x, 
                     z=z,
                     A_log=self.A_log,
                     D=self.D,
                     x_proj=self.x_proj,
                     dt_proj=self.dt_proj)

        if self.config.bidirectional:
            xz_b = xz.flip([1]) # (B, L, 2*ED)
            x_b, z_b = xz_b.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)
            x_b = x_b.transpose(1, 2) # (B, ED, L)
            x_b = self.conv1d_b(x_b)[:, :, :L] # depthwise convolution over time, with a short filter
            x_b = x_b.transpose(1, 2) # (B, L, ED)
            x_b = F.silu(x_b)
            y_b = self.ssm(x=x_b,
                           z=z_b,
                           A_log=self.A_log_b,
                           D=self.D_b,
                           x_proj=self.x_proj_b,
                           dt_proj=self.dt_proj_b)

        if self.config.use_cuda:
            if not self.config.bidirectional:
                return self.out_proj(y)
            else:
                if self.config.divide_output:
                    return self.out_proj((y + y_b.flip([1])) / 2)
                else:
                    return self.out_proj(y + y_b.flip([1]))
        
        z = F.silu(z)
        y = y * z
        if not self.config.bidirectional:
            return self.out_proj(y)
        else:
            z_b = F.silu(z_b)
            y_b = y_b * z_b
            if self.config.divide_output:
                return self.out_proj((y + y_b.flip([1])) / 2)
            else:
                return self.out_proj(y + y_b.flip([1]))
    
    def forward(self, x):
        if self.training and x.requires_grad and not self.config.use_cuda:
            return cp(self.inner_forward, x, use_reentrant=False)
        else:
            return self.inner_forward(x)
    
    def ssm(self, x, z, A_log, D, x_proj, dt_proj):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(A_log.float()) # (ED, N)
        D = D.float()

        deltaBC = x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h

#----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mixer = ViMBlock(config)
        
    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)
        return self.mixer(self.norm(x)) + x
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

#----------------------------------------------------------------------------

class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            in_channels=3,
            num_classes=1000,
            global_pool="token",
            embed_dim=192,
            depth=12,
            class_token=True,
            pos_embed="learn",
            no_embed_class=False,
            reg_tokens=0,
            pre_norm=False,
            fc_norm=None,
            dynamic_img_size=False,
            dynamic_img_pad=False,
            drop_rate=0.0,
            pos_drop_rate=0.0,
            patch_drop_rate=0.0,
            proj_drop_rate=0.0,
            drop_path_rate=0.0,
            weight_init="",
            fix_init=False,
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
        ):
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.SiLU
        
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        
        mamba_config = MambaConfig(
            d_model=embed_dim,
            n_layers=depth,
        )

        # Patch embedding
        self.patch_embed = embed_layer(
            img_size=(img_size, img_size),
            patch_size=(patch_size, patch_size),
            in_channels=in_channels,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        num_patches = self.patch_embed.num_patches + 1
        
        # CLS token
        self.cls_index = num_patches // 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        # Positional Embedding
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, num_patches, embed_dim))

        # Main blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(depth):
            block = Block(mamba_config)
            self.blocks += [block]
        self.norm = RMSNorm(embed_dim, eps=1e-5)

        # Output
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        # Initialize weights
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        
    def _pos_embed(self, x):
        pos_embed = self.pos_embed
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        x0, x1 = torch.chunk(x, chunks=2, dim=1)
        x = torch.cat([x0, cls_token, x1], dim=1)
        
        return x + pos_embed

    def forward_features(self, x):
        # Patchify: [B, 3, H, W] -> [B, N, C]
        x = self.patch_embed(x)

        # Positional embedding: [B, N, C] -> [B, N+1, C]
        x = self._pos_embed(x)

        # Main blocks: [B, N+1, C]
        for block in self.blocks:
            x = block(x)
        
        return x
    
    def forward_head(self, x):
        x = x[:, self.cls_index] # class token
        # x = self.fc_norm(x)
        # x = self.head_drop(x)
        x = self.head(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

#----------------------------------------------------------------------------