'''
    Implementation of the Mamba model in PyTorch.
    This version was ported almost as-is from the codes in:
    - https://github.com/alxndrTL/mamba.py/blob/main/mambapy/mamba.py
    to which all the credit goes.

    The only differences stems in the implementation of the Feature-wise Linear Modulation
    (FiLM) feature which was not present before.
'''

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pscan import pscan

@dataclass
class MambaFiLMConfig:
    d_model: int 
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 
    expand_factor: int = 2 
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" 
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False 

    mup: bool = False
    mup_base_width: float = 128 

    pscan: bool = True 
    use_cuda: bool = False 

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model 

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class MambaFiLM(nn.Module):
    def __init__(self, config: MambaFiLMConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, c):
        for layer in self.layers:
            x = layer(x, c)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaFiLMConfig):
        super().__init__()

        self.mixer = MambaBlockFiLM(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x, c):
        output = self.mixer(self.norm(x), c) + x
        return output

class MambaBlockFiLM(nn.Module):
    def __init__(self, config: MambaFiLMConfig):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)


        # ===== MODULATION PROJECTION LAYER INITIALISATION ===== #
        self.context_B = nn.Linear(config.d_model, 2 * config.d_state)
        self.context_C = nn.Linear(config.d_model, 2 * config.d_state)
        # ====================================================== #


        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

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
        inv_dt = dt + torch.log(-torch.expm1(-dt)) 
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
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

    def forward(self, x, c):
        _, L, _ = x.shape

        xz = self.in_proj(x) 
        x, z = xz.chunk(2, dim=-1) 

        x = x.transpose(1, 2) 
        x = self.conv1d(x)[:, :, :L] 
        x = x.transpose(1, 2) 

        x = F.silu(x)
        y = self.ssm(x, c, z)

        if self.config.use_cuda:
            output = self.out_proj(y) 
            return output 

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) 

        return output
    
    def ssm(self, x, c, z):

        A = -torch.exp(self.A_log.float()) 
        D = self.D.float()

        deltaBC = self.x_proj(x) 
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) 
    

        # ===== MODULATION PROJECTION ===== #
        gamma_B, beta_B = self.context_B(c).chunk(2, dim=-1)      # shape: (batch_size, d_state)
        gamma_C, beta_C = self.context_C(c).chunk(2, dim=-1)      # shape: (batch_size, d_state)
        # ================================= #


        # ===== MODULATION APPLICATION ===== #
        B = B * gamma_B.unsqueeze(1) + beta_B.unsqueeze(1)  # shape: (batch_size, seq_len, d_state)
        C = C * gamma_C.unsqueeze(1) + beta_C.unsqueeze(1)  # shape: (batch_size, seq_len, d_state)
        # =================================== #


        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) 
        
        if self.config.use_cuda:
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2) 
        
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):

        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) 

        BX = deltaB * (x.unsqueeze(-1)) 
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) 

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) 

        BX = deltaB * (x.unsqueeze(-1)) 

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) 
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) 

        y = (hs @ C.unsqueeze(-1)).squeeze(3) 

        y = y + D * x

        return y
    
    def step(self, x, cache):
        
        h, inputs = cache
        
        xz = self.in_proj(x) 
        x, z = xz.chunk(2, dim=1) 

        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] 

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) 

        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) 
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):

        A = -torch.exp(self.A_log.float()) 
        D = self.D.float()

        deltaBC = self.x_proj(x) 

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) 
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) 

        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) 

        BX = deltaB * (x.unsqueeze(-1)) 

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) 

        h = deltaA * h + BX 

        y = (h @ C.unsqueeze(-1)).squeeze(2) 

        y = y + D * x

        return y, h

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
    