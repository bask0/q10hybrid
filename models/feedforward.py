
import torch
from torch import nn
from torch import Tensor
from typing import Optional

from utils.torch_utils import get_activation


class FeedForward(nn.Module):
    """Implements a n-layered feed-forward neural networks.

    Implements a feed-forward model, each layer conisting of a linear,
    dropout and an activation layer. The dropout and actiavation of the
    last layer are optional (see `activation_last` and `dropout_last`).

    Args:
        num_inputs (int):
            Input dimensionality.
        num_outputs (int):
            Output dimensionality.
        num_hidden (int):
            Number of hidden units.
        num_layers (int):
            Number of fully-connected layers.
        dropout (float):
            Dropout applied after each layer, in range [0, 1).
        activation (str):
            Activation function name.
        activation_last (str, optional):
            If not `None`, this activation is applied after the last layer. Defaults to `None`.
        dropout_last (bool, optional):
            If `True`, the dropout is also applied after last layer. Defaults to `False`.
        layer_norm (bool):
            Wheter to use layer normalizations in all but the last layer. Defaults to `False`.

    """
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            activation: str,
            activation_last: Optional[torch.nn.Module] = None,
            dropout_last: bool = False,
            layer_norm: bool = False) -> None:

        super(FeedForward, self).__init__()

        activation = get_activation(activation)
        activation_last = get_activation(activation_last)

        in_sizes = [num_inputs] + [num_hidden] * (num_layers - 1)
        out_sizes = [num_hidden] * (num_layers - 1) + [num_outputs]

        layers = {}
        is_last = False
        for idx, (ni, no) in enumerate([(ni, no) for ni, no in
                                        zip(in_sizes, out_sizes)]):

            layer = nn.Linear(ni, no)

            if idx == num_layers - 1:
                is_last = True
            layers.update({f'linear{idx:02d}': layer})

            if not is_last:
                layers.update({f'dropout{idx:02d}': nn.Dropout(dropout)})
                if layer_norm:
                    layers.update({f'layer_norm{idx:02d}': nn.LayerNorm(no)})
                layers.update({f'activation{idx:02d}': activation})

            if is_last and dropout_last:
                layers.update({f'dropout{idx:02d}': nn.Dropout(dropout)})

            if is_last and activation_last is not None:
                layers.update({f'activation{idx:02d}': activation_last})

        self.model = nn.Sequential()

        for k, v in layers.items():
            self.model.add_module(k, v)

    def forward(self, x: Tensor) -> Tensor:
        """Mode lforward call.

        Args:
            x (Tensor):
                The input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return self.model(x)
