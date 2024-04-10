import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


#### WIRE NETWORK (Saragadam et. al. 2022) ####
class RealGaborLayer(nn.Module):
    """
    Implicit representation with Gabor nonlinearity

    in_features: Input features
    out_features: Output features
    bias: if True, enable bias for the linear operation
    is_first: Legacy SIREN parameter
    omega_0: Legacy SIREN parameter
    omega: Frequency of Gabor sinusoid term
    scale: Scaling of Gabor Gaussian term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 10.0,
        sigma0: float = 10.0,
        trainable: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input: torch.Tensor):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale**2))


class WireLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    in_features: Input features
    out_features: Output features
    bias: if True, enable bias for the linear operation
    is_first: Legacy SIREN parameter
    omega_0: Frequency of Gabor sinusoid term
    sigma0: Scaling of Gabor Gaussian term
    trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        sigma0: float = 10.0,
        trainable: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input: torch.Tensor):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class Wire2DLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    in_features: Input features
    out_features: Output features
    bias: if True, enable bias for the linear operation
    is_first: Legacy SIREN parameter
    omega_0: Frequency of Gabor sinusoid term
    sigma0: Scaling of Gabor Gaussian term
    trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        sigma0: float = 10.0,
        trainable: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        self.scale_orth = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)

        scale_x = lin
        scale_y = self.scale_orth(input)

        freq_term = torch.exp(1j * self.omega_0 * lin)

        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0 * self.scale_0 * arg)

        return freq_term * gauss_term


class Wire3DLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    in_features: Input features
    out_features: Output features
    bias: if True, enable bias for the linear operation
    is_first: Legacy SIREN parameter
    omega_0: Frequency of Gabor sinusoid term
    sigma0: Scaling of Gabor Gaussian term
    trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        sigma0: float = 10.0,
        trainable: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        self.scale_orth = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        self.scale_orth_2 = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)

        scale_x = lin
        scale_y = self.scale_orth(input)
        scale_z = self.scale_orth_2(input)

        freq_term = torch.exp(1j * self.omega_0 * lin)

        arg = scale_x.abs().square() + scale_y.abs().square() + scale_z.abs().square()
        gauss_term = torch.exp(-self.scale_0 * self.scale_0 * self.scale_0 * arg)

        return freq_term * gauss_term
