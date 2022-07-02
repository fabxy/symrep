"""This is a generic model for implementing SR3.

Given two identical models (with potential different initial parameters),
this model will manage two twin classes.
The loss function for training these is then:
`PredictiveLoss(f(...; theta1)) + RegularizationLoss(theta2) + SR3Loss(theta1, theta2)`.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .ghost_tuple import GhostTuple
from copy import deepcopy


class GhostWrapper(nn.Module):
    def __init__(self, model_constructor):
        super().__init__()

        if not isinstance(model_constructor, nn.Module):
            # Assume it's a function to construct one:
            model = model_constructor()
            ghost_model = model_constructor()
            assert isinstance(self.model, nn.Module)
        else:
            # Clone the model:
            model = model_constructor
            ghost_model = deepcopy(model_constructor)

        self._live_model = model
        self._ghost_model = ghost_model
        self.num_params = sum(p.numel() for p in self.parameters())
        self.reset_parameters()

    @property
    def live(self):
        return self._live_model
    
    @property
    def ghost(self):
        return self._ghost_model

    @property
    def model(self):
        return GhostTuple(live=self.live, ghost=self.ghost)

    def reset_parameters(self):
        """Copy parameters to twin"""
        with torch.no_grad():
            for p in self.model.parameters():
                p.ghost.data.copy_(p.live.data)

    def forward(self, x, **kwargs):
        return self.live(x, **kwargs)

