import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from liger_kernel.ops.utils import calculate_settings, ensure_contiguous

#TODO: Acceleration by fusing SpMM?

__all__ = [
    'LlamaMoC_triton',
    'LlamaMoC_autograd',
    'LlamaMoC_mixed'
]


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _fused_sparse_swiglu_fwd_kernel(
    # inputs
    gate_c_ptr, 
    v_ptr, 
    indice_ptr, 
    # outputs
    v_c_ptr,
    scaled_v_ptr,
    # shapes
    e: tl.constexpr, 
    act_channels: tl.constexpr,
    # meta
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)

    gate_c_ptr += program_id * act_channels
    v_ptr += program_id * e
    indice_ptr += program_id * act_channels
    v_c_ptr += program_id * act_channels
    scaled_v_ptr += program_id * e

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask_c = col_offsets < act_channels
    mask = col_offsets < e

    indice_row = tl.load(indice_ptr+col_offsets, mask=mask_c)
    gate_c = tl.load(gate_c_ptr+col_offsets, mask=mask_c).to(tl.float32)

    v_c = tl.load(v_ptr+indice_row, mask=mask_c)
    tl.store(v_c_ptr+col_offsets, v_c, mask=mask_c)

    scaled_v_c = silu(gate_c) * v_c
    tl.store(scaled_v_ptr+indice_row, scaled_v_c, mask=mask)


@triton.jit
def _fused_sparse_swiglu_bwd_kernel(
    # inputs
    grad_scaled_v_ptr,
    gate_c_ptr,
    indice_ptr,
    v_c_ptr,
    # outputs
    grad_gate_ptr, # Here we should get grad_gate (implement bwd of topk) rather than grad_gate_c
    grad_v_ptr,
    scaled_v_ptr, # by recomputation
    # shapes
    e: tl.constexpr, 
    act_channels: tl.constexpr,
    # meta
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)

    grad_scaled_v_ptr += program_id * e
    gate_c_ptr += program_id * act_channels
    indice_ptr += program_id * act_channels
    v_c_ptr += program_id * act_channels
    grad_gate_ptr += program_id * e
    grad_v_ptr += program_id * e
    scaled_v_ptr += program_id * e

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask_c = col_offsets < act_channels
    mask = col_offsets < e

    indice_row = tl.load(indice_ptr+col_offsets, mask=mask_c)
    gate_c = tl.load(gate_c_ptr+col_offsets, mask=mask_c).to(tl.float32)
    v_c = tl.load(v_c_ptr+col_offsets, mask=mask_c)

    sig = tl.sigmoid(gate_c)
    sil = sig * gate_c
    scaled_v_c = sil * v_c
    tl.store(scaled_v_ptr+indice_row, scaled_v_c, mask=mask)

    grad_scaled_v_c = tl.load(grad_scaled_v_ptr+indice_row, mask=mask_c)
    grad_v_c = grad_scaled_v_c * sil
    tl.store(grad_v_ptr+indice_row, grad_v_c, mask=mask)
    grad_gate_c = grad_scaled_v_c * v_c * (sil * (1-sig) + sig)
    tl.store(grad_gate_ptr+indice_row, grad_gate_c, mask=mask)


class FusedSparseSwiGLU(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, gate, v, down_proj_w, act_channels: int):
        ctx.shape = gate.shape
        ctx.act_channels = act_channels
        b, s, e = gate.shape

        gate_c, indice = torch.topk(gate, k=act_channels, dim=-1, sorted=False)
        del gate

        v_c = torch.empty_like(gate_c)
        scaled_v = torch.zeros_like(v) # zero padding

        BLOCK_SIZE, num_warps = calculate_settings(e) # Use a grid for each row

        _fused_sparse_swiglu_fwd_kernel[(b*s,)](
            # inputs
            gate_c,
            v,
            indice,
            # outputs
            v_c,
            scaled_v,
            # shapes
            e,
            act_channels,
            # meta
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(gate_c, indice.to(torch.uint16), v_c, down_proj_w)
        return scaled_v @ down_proj_w.t()
    
    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        gate_c, indice, v_c, down_proj_w = ctx.saved_tensors
        indice = indice.to(torch.int64) # Only int64 can be used as indices
        b, s, e = ctx.shape

        grad_scaled_v = grad_output @ down_proj_w
        del down_proj_w

        grad_gate = torch.zeros(ctx.shape, device=gate_c.device, dtype=gate_c.dtype)
        grad_v = torch.zeros_like(grad_gate)
        scaled_v = torch.zeros_like(grad_gate) # gain by recomputation

        BLOCK_SIZE, num_warps = calculate_settings(e) # Use a grid for each row

        _fused_sparse_swiglu_bwd_kernel[(b*s,)](
            # inputs
            grad_scaled_v,
            gate_c,
            indice,
            v_c,
            # outputs
            grad_gate,
            grad_v,
            scaled_v,
            # shapes
            e,
            ctx.act_channels,
            # meta
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        grad_down_proj = torch.einsum('bsd, bse -> de', grad_output, scaled_v) # faster than mm+sum
        return grad_gate, grad_v, grad_down_proj, None
    

class LlamaMoC_triton(nn.Module): 
    """
    LlamaMoC implemented on Triton fused kernel, fast and memory-efficient, 
    but the result is a little bit inaccurate compared to PyTorch's native implementation.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        act_channels: int,
    ):
        assert hidden_act == 'silu', 'Currenly only support LLaMA-style SwiGLU!'
        assert intermediate_size <= 2**16, 'Intermediate_size must be no more than 2^16 to save indices as uint16!'
        
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False) # do not support bias
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        self.act_channels = act_channels
        self.hidden_act = hidden_act

    def forward(self, x):
        gate = self.gate_proj(x)
        v = self.up_proj(x)
        return FusedSparseSwiGLU.apply(gate, v, self.down_proj.weight, self.act_channels)


class LlamaMoC_autograd(nn.Module): # Not memory-efficient
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        act_channels: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.SiLU()
        self.act_channels = act_channels

    def forward(self, x):
        gate = self.gate_proj(x)
        gate_c, indices = torch.topk(gate, self.act_channels, dim=-1, largest=True, sorted=False)
        act_c = self.act_fn(gate_c)
        v_c = self.up_proj(x).gather(-1, indices)
        scaled_v_c = act_c * v_c
        scaled_v = torch.zeros_like(gate).scatter_(-1, indices, scaled_v_c)
        return self.down_proj(scaled_v)

    
class LlamaMixedMoC(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, gate, v, proj_w, act_channels: int):
        ctx.shape = gate.shape # (b, s, e)
        gate_c, indices = torch.topk(gate, act_channels, dim=-1, largest=True, sorted=False)
        act_c = torch.ops.aten.silu(gate_c) 
        v_c = v.gather(-1, indices)
        scaled_v_c = act_c * v_c
        scaled_v = torch.zeros(ctx.shape, device=scaled_v_c.device, dtype=scaled_v_c.dtype).scatter_(-1, indices, scaled_v_c)
        ctx.save_for_backward(proj_w, gate_c, act_c, v_c, scaled_v_c, indices.to(torch.uint16))
        return torch.matmul(scaled_v, proj_w.t())
    
    @staticmethod
    def backward(ctx, grad_output):
        proj_w, gate_c, act_c, v_c, scaled_v_c, indices = ctx.saved_tensors
        indices = indices.to(torch.int64)
        grad_scaled_v = torch.matmul(grad_output, proj_w)
        grad_scaled_v_c = grad_scaled_v.gather(-1, indices)
        zero = torch.zeros(ctx.shape, device=scaled_v_c.device, dtype=scaled_v_c.dtype)
        scaled_v = zero.scatter(-1, indices, scaled_v_c)
        grad_proj_w = grad_output.view(-1, grad_output.shape[-1]).t().mm(scaled_v.view(-1, scaled_v.shape[-1]))
        grad_v_c = act_c * grad_scaled_v_c
        grad_act_c = v_c * grad_scaled_v_c
        grad_gate_c = torch.ops.aten.silu_backward(grad_act_c, gate_c)
        grad_gate = zero.scatter(-1, indices, grad_gate_c)
        grad_v = zero.scatter(-1, indices, grad_v_c)
        return grad_gate, grad_v, grad_proj_w, None


class LlamaMoC_mixed(nn.Module): 
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        act_channels: int,
    ):
        assert hidden_act == 'silu', 'Currenly only support LLaMA-stype SwiGLU'
        assert intermediate_size < 2**16, 'Intermediate_size must be smaller than 2^16 to save indices as uint16!'
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False) # do not support bias
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # For simplicity, assume no dropout (common for PT).
        
        self.act_channels = act_channels
        self.hidden_act = hidden_act


    def forward(self, x):
        gate = self.gate_proj(x)
        v = self.up_proj(x)
        return LlamaMixedMoC.apply(gate, v, self.down_proj.weight, self.act_channels)