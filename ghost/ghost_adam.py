"""The GhostAdam optimizer for regularized optimization."""

import torch
from torch import nn
from torch import optim
from .ghost_wrapper import GhostWrapper
from .ghost_tuple import GhostTuple


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

    def get_param_differences_and_scales(self):
        differences = []
        combined_scales = []

        param_group = GhostTuple(live=self.param_groups[0], ghost=self.param_groups[1])

        beta2 = param_group["betas"][0]

        for p in param_group["params"]:

            state = GhostTuple(
                live=self.state[p.live],
                ghost=self.state[p.ghost],
            )

            if len(state.live) == 0:
                continue
            elif state.live["step"] == 0:
                continue

            if "step" not in state.ghost or state["step"].live != state["step"].ghost:
                raise ValueError(
                    "You have not trained the ghost model as many steps "
                    "as the primary model."
                )

            step = state["step"]

            bias_correction2 = 1 - beta2**step
            v_t = state["exp_avg_sq"]  # In Adam paper, this is v_t
            hat_v_t = v_t / bias_correction2

            combined_hat_v_t = hat_v_t.live + hat_v_t.ghost

            # Adam: update = -lr * hat_m_t / (sqrt(hat_v_t) + eps)
            combined_scales.append(1 / (torch.sqrt(combined_hat_v_t) + self.eps))
            differences.append(p.live - p.ghost)

        return differences, combined_scales

    def ghost_loss_backward(self):
        differences, combined_scales = self.get_param_differences_and_scales()

        if len(combined_scales) == 0:
            return

        sum_scales = sum(s.sum() for s in combined_scales)
        num_scales = sum(s.numel() for s in combined_scales)
        scale_normalization = sum_scales / num_scales

        # Make the average scale equal to 1:
        combined_scales = [s / scale_normalization for s in combined_scales]

        distance = torch.stack(
            [(d.pow(2) * s).sum() for (d, s) in zip(differences, combined_scales)]
        ).sum()
        ghost_loss = self.ghost_coeff * distance
        self.ghost_loss = ghost_loss.item()
        ghost_loss.backward()

    def ensure_all_params_have_gradient(self):
        param_group = GhostTuple(live=self.param_groups[0], ghost=self.param_groups[1])

        all_p = []
        for p in param_group["params"]:
            all_p.append(p.live.sum() * 0.0 + p.ghost.sum() * 0.0)

        torch.stack(all_p).sum().backward()

    def step(self):
        """Normal Adam step, but we include an SR3 loss."""
        self.ghost_loss_backward()
        self.ensure_all_params_have_gradient()

        super().step()
