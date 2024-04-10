from models.wire import RealGaborLayer, Wire2DLayer, Wire3DLayer, WireLayer
from models.siren import SineLayer
from models.relu import ReluLayer
from torch import nn
import torch
import numpy as np


class INR(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30.0,
        sigma0: float = 10.0,
        inr: str = "wire",
        skip_conn: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()

        self.net = []
        self.domain_dim = in_features
        self.range_dim = out_features
        self.inr = inr
        self.skip_conn = skip_conn
        self.batchnorm = batchnorm

        if inr == "wire":
            self.nn = WireLayer
            dtype = torch.cfloat
            bias = True
            trainable = False
        elif inr == "siren":
            self.nn = SineLayer
            dtype = torch.float
            bias = False
            trainable = True
        elif inr == "relu":
            self.nn = ReluLayer
            dtype = torch.float
            bias = False
            trainable = True
        else:
            raise Exception(
                f"Invalid inr selected: {inr}. Valid options are siren and wire."
            )

        self.net.append(
            self.nn(
                in_features,
                hidden_features,
                is_first=True,
                trainable=trainable,
                omega_0=first_omega_0,
                sigma0=sigma0,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nn(
                    hidden_features,
                    hidden_features,
                    omega_0=hidden_omega_0,
                    sigma0=sigma0,
                    batchnorm=self.batchnorm,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype)
        if inr == "siren":
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor):
        """
        coords: is a tensor of shape (B, 3) where B is the number of points and 3 is the dimension of the domain.
        """
        coords = coords.to(torch.float)

        if self.skip_conn:
            first_output = self.net[:1](coords)
            second_output = self.net[1:-2](first_output)
            second_input = first_output + second_output  # skip connection
            output = self.net[-2:](second_input)
        else:
            output = self.net(coords)

        if self.inr == "wire":
            output = output.real
        if not self.training:
            output = output.to(torch.float)

        return output
