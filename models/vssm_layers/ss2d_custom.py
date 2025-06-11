from functools import partial
import math
import torch
import torch.nn as nn

from .mlp import Linear
from .layer_norm import LayerNorm
from .csm_triton_custom import cross_scan_fn, cross_merge_fn
from .csm import selective_scan_fn as selective_scan
# from .selective_scan_trt import selective_scan

#----------------------------------------------------------------------------

def checkpostfix(tag, value):
    ret = value[-len(tag):] == tag
    if ret:
        value = value[:-len(tag)]
    return ret, value

#----------------------------------------------------------------------------

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

#----------------------------------------------------------------------------

class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=1,
            ssm_ratio=1.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=False,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v05_noz",
            channel_first=True,
            scan_route=0,
            # ======================
            **kwargs,
        ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        assert scan_route in [0, 1, 2, 3]
        self.scan_route = scan_route
        self.k_group = 1
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        
        # tags for forward_type ============================== (v05_noz -> v05, disable_z)
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
        
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            # v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            # v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            # v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            # v04=partial(self.forward_corev2, force_fp32=False), # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            # v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            # v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            # v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3), # debug
            # ===============================
            # v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            # v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        
        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        
        # x proj ============================
        self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group, bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False, channel_first=True)
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        else:
            raise NotImplementedError
        # elif initialize in ["v1"]:
        #     # simple init dt_projs, A_logs, Ds
        #     self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        #     self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        #     self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
        #     self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        # elif initialize in ["v2"]:
        #     # simple init dt_projs, A_logs, Ds
        #     self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        #     self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        #     self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
        #     self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias
    
    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            raise NotImplementedError
        elif out_norm_dwconv3:
            raise NotImplementedError
        elif out_norm_softmax:
            raise NotImplementedError
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type
    
    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "unidi",
        scan_route = 0,
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        assert _scan_mode == 1 and scan_route in [0, 1, 2, 3]
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        # def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        #     return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if True:
            # [B, C, h, w] -> [B, K=4, C, L]
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, scan_route=scan_route, force_torch=scan_force_torch)
            xs = xs.sum(dim=1, keepdim=True) / 4
            x_dbl = self.x_proj(xs.view(B, -1, L))
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
            Ds = self.Ds.to(torch.float) # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            # [B, K=4, C, h, w] -> [B, C, L]
            ys = ys.repeat(1, 4, 1, 1, 1) / 4
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, scan_route=scan_route, force_torch=scan_force_torch)
            
            # if getattr(self, "__DEBUG__", False):
            #     setattr(self, "__data__", dict(
            #         A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
            #         us=xs, dts=dts, delta_bias=delta_bias,
            #         ys=ys, y=y, H=H, W=W,
            #     ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)

        return y.to(x.dtype)
    
    def forward(self, x):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, scan_route=self.scan_route)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

#----------------------------------------------------------------------------

if __name__ == "__main__":
    forward_type = "v05_noz"
    disable_force32, forward_type = checkpostfix("_no32", forward_type)
    oact, forward_type = checkpostfix("_oact", forward_type)
    disable_z, forward_type = checkpostfix("_noz", forward_type)
    disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
    