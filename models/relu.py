from torch import nn
import torch


class ReluLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        sigma0: float = 10.0,
        trainable: bool = True,
        batchnorm: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        return self.relu(self.linear(input))
