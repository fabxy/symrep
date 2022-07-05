"""The GhostAdam optimizer for regularized optimization."""

import torch
from torch import nn
from torch import optim
from .ghost_wrapper import GhostWrapper
from .ghost_tuple import GhostTuple
import warnings


class GhostAdam(optim.Adam):
    """The GhostAdam optimizer for regularized optimization.

    This is a variant of Adam that uses ghost regularization.
    """

    def __init__(
        self,
        parameters_or_model,
        ghost_parameters=None,
        *,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        ghost_coeff=1e-6,
        ghost_scaled=True,
    ):
        if isinstance(parameters_or_model, nn.Module):
            if ghost_parameters is not None:
                raise ValueError(
                    "Do not specify ghost_parameters if you pass in a model."
                )
            if not isinstance(parameters_or_model, GhostWrapper):
                raise ValueError(
                    "parameters_or_model must be an GhostWrapper if you pass in a model."
                )
            model = parameters_or_model
            parameters = GhostTuple(
                live=model.live.parameters(), ghost=model.ghost.parameters()
            )
        else:
            parameters = GhostTuple(live=parameters_or_model, ghost=ghost_parameters)
            for p in parameters:
                if p.live.shape != p.ghost.shape:
                    raise ValueError(
                        "Both sets of parameters must have the same shape and order."
                    )

        self.eps = eps
        self.ghost_coeff = ghost_coeff
        self.ghost_loss = None
        self.scaled = ghost_scaled
        self.ghost_step = 0

        super().__init__(
            [
                {"params": parameters.live, "weight_decay": 0},
                {"params": parameters.ghost},
            ],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @property
    def ghost_tuple_param_group(self):
        return GhostTuple(live=self.param_groups[0], ghost=self.param_groups[1])

    def get_param_distance(self, scaled=True):
        differences = []
        combined_scales = []

        param_group = self.ghost_tuple_param_group

        beta2 = param_group["betas"][0]
        distances = []
        scales = []
        numel = 0

        for p in param_group["params"]:

            state = GhostTuple(
                live=self.state[p.live],
                ghost=self.state[p.ghost],
            )

            if len(state.live) == 0:
                continue
            elif state.live["step"] == 0:
                continue

            if "step" not in state.ghost:
                raise ValueError(
                    "You have not trained the ghost model as many steps "
                    "as the primary model."
                )

            # Adam: update = -lr * hat_m_t / (sqrt(hat_v_t) + eps)
            if scaled:
                step = state["step"]
                bias_correction2 = 1 - beta2**step
                v_t = state["exp_avg_sq"]  # In Adam paper, this is v_t
                hat_v_t = v_t / bias_correction2
                combined_hat_v_t = hat_v_t.live + hat_v_t.ghost
                s = 1 / (torch.sqrt(combined_hat_v_t) + self.eps)
            else:
                s = torch.ones_like(p.live.data)

            distances.append(
                ((p.live - p.ghost).pow(2) * s).sum()
            )
            scales.append(s.sum())
            numel += p.live.numel()

        if len(scales) == 0:
            if self.ghost_step == 1:
                warnings.warn(
                    "It seems you are not computing gradients for the ghost "
                    "model. This warning will be silenced."
                )
            # No parameters have gradients yet.
            return

        scale_normalization = torch.sum(torch.stack(scales)) / numel
        distance = torch.sum(torch.stack(distances))

        scaled_distance = distance / scale_normalization
        return scaled_distance

    def ghost_loss_backward(self):
        scaled_distance = self.get_param_distance(scaled=self.scaled)

        if scaled_distance is None:
            return

        ghost_loss = self.ghost_coeff * scaled_distance
        self.ghost_loss = ghost_loss.item()
        ghost_loss.backward()

    def ensure_all_params_have_gradient(self):
        param_group = self.ghost_tuple_param_group

        all_p = []
        for p in param_group["params"]:
            # Ensure parameters have a gradient:
            if p.live.grad is None:
                (p.live.sum() * 0.0).backward()
            if p.ghost.grad is None:
                (p.ghost.sum() * 0.0).backward()

    def step(self):
        """Normal Adam step, but we include an SR3 loss."""
        self.ensure_all_params_have_gradient()

        if self.ghost_step > 0:
            self.ghost_loss_backward()

        super(GhostAdam, self).step()
        self.ghost_step += 1
