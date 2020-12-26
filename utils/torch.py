
from torch import nn
from typing import Union
from torch import Tensor


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


def is_flat(x: Tensor) -> bool:
    """Checks if tensor is in the flat format (batch, sequence x num_features)

    Args:
        x (Tensor): the tensor.
    """
    return x.ndim == 3


def is_sequence(x: Tensor) -> bool:
    """Checks if tensor is in the sequence format (batch, sequence, num_features)

    Args:
        x (Tensor): the tensor.
    """
    return x.ndim == 2


def seq2flat(x: Tensor) -> Tensor:
    """Reshapes tensor from sequence format to flat format.

    Sequence format: (batch, sequence, features)
    Flat format: (batch, sequence x features)

    Args:
        x (Tensor): a tensor in the sequence format (batch, sequence, features).

    Returns:
        Tensor: the transformed tensor in flat format (batch, sequence x features).
    """

    if not is_sequence(x):
        raise ValueError(
            'attempt to reshape tensor from sequence format to flat format failed. ',
            f'Excepted input tensor with 3 dimensions, got {x.ndim}.'
        )

    return x.flatten(start_dim=1)


def flat2seq(x: Tensor, num_features: int) -> Tensor:
    """Reshapes tensor from flat format to sequence format.

    Flat format: (batch, sequence x features)
    Sequence format: (batch, sequence, features)

    Args:
        x (Tensor): a tensor in the flat format (batch, sequence x features).
        num_features (int): number of features (last dimension) of the output tensor.

    Returns:
        Tensor: the transformed tensor in sequence format (batch, seq, features).
    """

    if not is_flat(x):
        raise ValueError(
            'attempt to reshape tensor from flat format to sequence format failed. ',
            f'Excepted input tensor with 2 dimensions, got {x.ndim}.'
        )

    return x.view(x.shape[0], -1, num_features)
