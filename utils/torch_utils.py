
from torch import nn
from typing import Union


def get_activation(activation: Union[str, None]) -> nn.Module:
    """Get PyTorch activation function by string query.

    Args:
        activation (str, None): activation function, one of `relu`, `leakyrelu`, `selu`, `sigmoid`, `softplus`,
            `tanh`, `identity` (aka `linear` or `none`). If `None` is passed, `None` is returned.

    Raises:
        ValueError: if activation not found.

    Returns:
        PyTorch activation or None: activation function
    """

    if activation is None:
        return None

    a = activation.lower()

    if a == 'linear' or a == 'none':
        a = 'identity'

    activations = dict(
        relu=nn.ReLU,
        leakyrelu=nn.LeakyReLU,
        selu=nn.SELU,
        signoid=nn.Sigmoid,
        softplus=nn.Softplus,
        tanh=nn.Tanh,
        identity=nn.Identity
    )

    if a in activations:
        return activations[a]()
    else:
        choices = ', '.join(list(activations.keys()))
        raise ValueError(
            f'activation `{activation}` not found, chose one of: {choices}.'
        )
