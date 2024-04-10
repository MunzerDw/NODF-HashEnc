import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


#### SIREN NETWORK (Sitzmann et. al. 2020, NeurIPS) ####
class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
        sigma0: float = 10.0,
        trainable: bool = True,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.batchnorm = batchnorm

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

        if self.batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(out_features)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: torch.Tensor):
        result = torch.sin(self.omega_0 * self.linear(input))
        if self.batchnorm:
            result = self.batchnorm_layer(result)
        return result

    def forward_with_intermediate(self, input: torch.Tensor):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


def init_weights_normal(m: nn.Linear):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


def init_weights_selu(m: nn.Linear):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m: nn.Linear):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            nn.init.normal_(
                m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input)
            )


def init_weights_xavier(m: nn.Linear):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.xavier_normal_(m.weight)


def sine_init(m: nn.Linear):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m: nn.Linear):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
