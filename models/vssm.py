from collections import OrderedDict
import torch
import torch.nn as nn

from .vssm_layers.layer_norm import LayerNorm
from .vssm_layers.permute import Permute
from .vssm_layers.vss import VSSBlock
from .vssm_layers.ss2d import SS2D

from .weight_init import trunc_normal_

#----------------------------------------------------------------------------

class VSSM(nn.Module):
    channel_first = True
    
    def __init__(
            self,
            img_size=32,
            patch_size=4,
            num_classes=10,
            depths=[2, 2, 8, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.2,
            patch_norm=True,
            norm_layer="ln2d",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
            # =========================
            posembed=False,
            _SS2D=SS2D,
            tensorrt=False,
            # =========================
            **kwargs,
        ):
        super().__init__()
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        _ACTLAYERS = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
        }
        ssm_act_layer = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        
        self.patch_embed = self._make_patch_embed(
            in_channels=3,
            embed_dim=dims[0],
            patch_size=patch_size,
            patch_norm=patch_norm,
            channel_first=True,
            version=patchembed_version,
        )
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = self._make_downsample(
                dim=self.dims[i_layer],
                out_dim=self.dims[i_layer+1],
                channel_first=True,
                version=downsample_version,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
            
            layer = self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                use_checkpoint=use_checkpoint,
                downsample=downsample,
                channel_first=True,
                # =========================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =========================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =========================
                _SS2D=_SS2D,
            )
            self.layers += [layer]
        
        self.classifier = nn.Sequential(
            OrderedDict(
                norm=LayerNorm(self.num_features, channel_first=self.channel_first), # B,H,W,C
                permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
                avgpool=nn.AdaptiveAvgPool2d(1),
                flatten=nn.Flatten(1),
                head=nn.Linear(self.num_features, num_classes),
            )
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def _make_patch_embed(in_channels, embed_dim, patch_size, patch_norm=True, channel_first=True, version="v2"):
        if version == "v2":
            stride = patch_size // 2
            kernel_size = stride + 1
            padding = 1
            return nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
                LayerNorm(embed_dim // 2, channel_first=True),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first),
            )
        else:
            raise NotImplementedError
    
    @staticmethod
    def _make_downsample(dim, out_dim, norm=True, channel_first=True, version="v3"):
        if version == "v3":
            return nn.Sequential(
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first),
            )
        else:
            raise NotImplementedError
    
    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            downsample=nn.Identity(),
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
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # =========================
            **kwargs,
        ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks += [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    channel_first=channel_first,
                    # =========================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =========================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    # =========================
                    use_checkpoint=use_checkpoint,
                    post_norm=False,
                ),
            ]
        
        return nn.Sequential(
            OrderedDict(
                blocks=nn.Sequential(*blocks),
                downsample=downsample,
            )
        )
    
    def forward(self, x):
        # [B, 3, H, W] -> [B, C0, h0, w0]
        x = self.patch_embed(x)
        
        # [B, C0, h0, w0], [B, C1, h1, w1], ...
        for layer in self.layers:
            x = layer(x)
        
        # [B, C3, h3, w3] -> [B, cls]
        x = self.classifier(x)
        
        return x

#----------------------------------------------------------------------------