import numpy as np
import einops as E

import torch as torch
from torch import nn as nn
from torch.functional import F as F

class SparseJacobianNN(nn.Module):
    """
    A Multi-layer perceptron that computes its outputs
        completely independently from one another. The
        network, when the net's KL divergence is minimized
        (`.sparsifying_loss()`), will result in a sparse Jacobian."""

    def __init__(
        self,
        n_in,
        n_out,
        hidden=100,
        nlayers=2,
        act=F.relu,
        approximate_l0_with=lambda x, dim: x.abs().sum(dim=dim),
        approximate_max_with=lambda x, dim: x.pow(2).sum(dim=dim).pow(0.5),
        norm=None,
    ):
        """
        Args:
            n_in: number of input dimensions
            n_out: number of output dimensions
            hidden: number of hidden units
            nlayers: number of hidden layers
            act: activation function
            a1: weight encouraging few output features
            a2: weight encouraging few dependencies for each output
            approximate_l0_with: differentiable function that approximates the L0 norm of a tensor
            approximate_max_with: differentiable function that approximates the max of a tensor
            norm: row-wise mask normalization
        """
        super(self.__class__, self).__init__()
        self.n_out = n_out
        self.alpha = nn.Parameter(torch.randn(n_out, n_in))
        self.nlayers = nlayers
        self.act = act
        self.l0_func = approximate_l0_with
        self.max_func = approximate_max_with
        self.w = nn.ParameterList(
            [init_weights_3d(n_out, n_in, hidden)]
            + [init_weights_3d(n_out, hidden, hidden) for i in range(nlayers)]
            + [init_weights_3d(n_out, hidden, 1)]
        )
        self.b = nn.ParameterList(
            [init_weights_2d(n_out, hidden)]
            + [init_weights_2d(n_out, hidden) for i in range(nlayers)]
            + [init_weights_2d(n_out, 1)]
        )
        if norm == "softmax":
            self.norm = F.softmax
        else:
            self.norm = norm

    def apply_layer(self, i, x):
        x = torch.einsum("oih,boi->boh", self.w[i], x)
        x = x + self.b[i][None, :, :]
        return x

    def get_masked_input(self, x):
        tiled = E.repeat(x, "batch n_in -> batch n_out n_in", n_out=self.n_out)

        if self.norm:
            self.alpha_n = self.norm(self.alpha.abs(), dim=1)
        else:
            self.alpha_n = self.alpha

        return tiled * self.alpha_n.unsqueeze(0)

    def sparsifying_loss(self, a1, a2):
        importances = self.alpha  # [n_out, n_in]
        few_latents = self.l0_func(importances, dim=1).sum()
        few_dependencies = self.max_func(self.l0_func(importances, dim=1), dim=0)
        # few_latents = self.alpha.abs().sum()
        # few_dependencies = self.alpha.pow(2).sum().pow(-1)
        return a1 * few_latents + a2 * few_dependencies

    def entropy(self):
        return -(self.alpha_n * self.alpha_n.log2()).sum(dim=1)

    def entropy_loss(self, e1):
        return e1 * self.entropy().pow(2).sum()

    def forward(self, x):
        xtilde = self.get_masked_input(x)
        for i in range(self.nlayers + 1):
            xtilde = self.act(self.apply_layer(i, xtilde))

        out = self.apply_layer(self.nlayers + 1, xtilde)
        return torch.squeeze(out, dim=-1)


def init_weights_3d(n1, n2, n3):
    xavier = np.sqrt(6 / (n2 + n3))
    return nn.Parameter(torch.randn(n1, n2, n3) * xavier)


def init_weights_2d(n1, n2):
    return nn.Parameter(torch.zeros(n1, n2))
