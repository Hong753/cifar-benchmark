from typing import Tuple, List

import torch
import torch_tensorrt
import tensorrt.plugin as trtp

import selective_scan_cuda_core

#----------------------------------------------------------------------------

# try:
#     import selective_scan_cuda_core
# except Exception as e:
#     ...
#     print("WARNING: can not import selective_scan_cuda_core.", flush=True)
#     print(e, flush=True)
#     raise RuntimeError()

# @torch.library.custom_op("custom_ops::selective_scan", mutates_args=())
# def selective_scan(
#         u: torch.Tensor,
#         delta: torch.Tensor,
#         A: torch.Tensor,
#         B: torch.Tensor,
#         C: torch.Tensor,
#         D: torch.Tensor,
#         delta_bias: torch.Tensor,
#         delta_softplus: bool,
#     ) -> torch.Tensor:
#     out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
#     return out

# @torch.library.register_fake("custom_ops::selective_scan")
# def _selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
#     batch, d_inner, seqlen = u.shape
#     out_shape = [batch, d_inner, seqlen]
#     return u.new_zeros(out_shape)

#----------------------------------------------------------------------------
# Selective scan

@torch.library.custom_op("torchtrt_ex::selective_scan", mutates_args=())
def selective_scan(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        delta_bias: torch.Tensor,
        delta_softplus: bool,
    ) -> torch.Tensor:
    out, x, *rest = selective_scan_cuda_core.fwd(
        u,
        delta,
        A,
        B,
        C,
        D,
        delta_bias,
        delta_softplus,
        1,
    )
    return out

@torch.library.register_fake("torchtrt_ex::selective_scan")
def _(u, delta, A, B, C, D, delta_bias, delta_softplus):
    return u

# @trtp.register("torchtrt_ex::selective_scan")
# def _(
#         u: trtp.TensorDesc,
#         delta: trtp.TensorDesc,
#         A: trtp.TensorDesc,
#         B: trtp.TensorDesc,
#         C: trtp.TensorDesc,
#         D: trtp.TensorDesc,
#         delta_bias: trtp.TensorDesc,
#         delta_softplus: bool,
#     ) -> Tuple[trtp.TensorDesc]:
#     return u.like()

# @trtp.impl("torchtrt_ex::selective_scan")
# def _(
#         u: trtp.Tensor,
#         delta: trtp.Tensor,
#         A: trtp.Tensor,
#         B: trtp.Tensor,
#         C: trtp.Tensor,
#         D: trtp.Tensor,
#         delta_bias: trtp.Tensor,
#         delta_softplus: bool,
#         outputs: Tuple[trtp.Tensor],
#         stream: int,
#     ):
#     with torch.cuda.stream(torch.cuda.ExternalStream(stream)):    
#         u_t = torch.as_tensor(u, device="cuda")
#         delta_t = torch.as_tensor(delta, device="cuda")
#         A_t = torch.as_tensor(A, device="cuda")
#         B_t = torch.as_tensor(B, device="cuda")
#         C_t = torch.as_tensor(C, device="cuda")
#         D_t = torch.as_tensor(D, device="cuda")
#         delta_bias_t = torch.as_tensor(delta_bias, device="cuda")
#         out_t = torch.as_tensor(outputs[0], device="cuda")
        
#         out, x, *rest = selective_scan_cuda_core.fwd(
#             u_t,
#             delta_t,
#             A_t,
#             B_t,
#             C_t,
#             D_t,
#             delta_bias_t,
#             delta_softplus,
#             1,
#         )
#         out_t.copy_(out)

# @trtp.autotune("torchtrt_ex::selective_scan")
# def _(
#     u: trtp.TensorDesc,
#     delta: trtp.TensorDesc,
#     A: trtp.TensorDesc,
#     B: trtp.TensorDesc,
#     C: trtp.TensorDesc,
#     D: trtp.TensorDesc,
#     delta_bias: trtp.TensorDesc,
#     delta_softplus: bool,
#     outputs: Tuple[trtp.TensorDesc],
# ) -> List[trtp.AutoTuneCombination]:
#     return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16, FP32|FP16, FP32|FP16, FP32|FP16, FP32|FP16, FP32|FP16, FP32|FP16", "LINEAR")]

# torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter("torchtrt_ex::selective_scan")

#----------------------------------------------------------------------------