import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .drop_path import DropPath
from .layer_norm import LayerNorm
from .ss2d_custom import SS2D
from .mlp import Mlp

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

#----------------------------------------------------------------------------

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            drop_path,
            channel_first=True,
            # =========================
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate="0.0",
            # =========================
            use_checkpoint=False,
            post_norm=False,
            # =========================
            **kwargs,
        ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        
        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            ss2d_kwargs = {
                # basic dims ===========
                "d_model": hidden_dim,
                "d_state": ssm_d_state,
                "ssm_ratio": ssm_ratio,
                "dt_rank": ssm_dt_rank,
                "act_layer": ssm_act_layer,
                # dwconv ===============
                "d_conv": ssm_conv, # < 2 means no conv 
                "conv_bias": ssm_conv_bias,
                # ======================
                "dropout": ssm_drop_rate,
                "bias": False,
                # dt init ==============
                "dt_min": 0.001,
                "dt_max": 0.1,
                "dt_init": "random",
                "dt_scale": 1.0,
                "dt_init_floor": 1e-4,
                "initialize": ssm_init,
                # ======================
                "forward_type": forward_type,
                "channel_first": channel_first,
                # ======================
            }
            self.op1 = SS2D(scan_route=0, **ss2d_kwargs)
            self.op2 = SS2D(scan_route=1, **ss2d_kwargs)
            self.op3 = SS2D(scan_route=2, **ss2d_kwargs)
            self.op4 = SS2D(scan_route=3, **ss2d_kwargs)
            
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channel_first=channel_first,
            )
    def op(self, x):
        x1 = x2 = x3 = x4 = x
        x_mamba1 = self.op1(x1.contiguous())
        x_mamba2 = self.op2(x2.contiguous())
        x_mamba3 = self.op3(x3.contiguous())
        x_mamba4 = self.op4(x4.contiguous())
        return x_mamba1 + x_mamba2 + x_mamba3 + x_mamba4
    
    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

#----------------------------------------------------------------------------