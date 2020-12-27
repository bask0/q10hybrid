"""Temporal Concolutional Network for time-series

Adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py under the
following license:

MIT License

Copyright (c) 2018 CMU Locus Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from typing import Dict

from utils.pl_utils import LightningNet
from utils.torch_utils import Transform


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float = 0.2) -> None:

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1,
                                 self.conv2, self.chomp2, self.relu2,
                                 self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(LightningNet):
    """Implements a Temporal Convolutional Network (TCN).

    https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

    Shapes:
        Input:  (batch_size, input_size, sequence_length)
        Output: (batch_size, num_channels[-1], sequence_length)

    Args:
        training_config (Dict): the training configuration passed to the superclass `LightningNet`.
        num_inputs (int): the mumber of input features.
        num_intputs (int): the number of outputs.
        num_hidden (int): the hidden size (intermediate channel sizes) of the layers.
        kernel_size (int): the kernel size. Defaults to 4.
        num_layers (int): the number of stacked layers. Defaults to 2.
        dropout (float): a float value in the range [0, 1) that defines the dropout probability. Defaults to 0.0.
    """

    def __init__(
            self,
            training_config: Dict,
            num_inputs: int,
            num_outputs: int,
            num_hidden: int,
            kernel_size: int = 4,
            num_layers: int = 2,
            dropout: float = 0.0,) -> None:

        super(TemporalConvNet, self).__init__(**training_config)

        # Used to calculate receptive field (`self.receptive_field_size`).
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden
            layers += [
                TemporalBlock(
                    in_channels,
                    num_hidden,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout)
            ]

        transform = Transform(transform_fun=lambda x: x.permute(0, 2, 1))

        linear = nn.Linear(num_hidden, num_outputs)

        self.network = nn.Sequential(
            *layers,  # -> (batch, num_hidden, seq)
            transform,  # -> (batch, seq, num_hidden)
            linear,  # -> (batch, seq, num_out)
            transform,  # -> (batch, num_out, seq)
        )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run data through the model.

        Args:
            x (torch.Tensor): the input data with shape (batch, num_inputs, seq).

        Returns:
            torch.Tensor: the model output with shape (batch, seq, num_outputs).
        """
        return self.network(x)

    def receptive_field_size(self) -> int:
        """Returns the receptive field of the Module.

        The receptive field (number of steps the model looks back) of the model depends
        on the number of layers and the kernel size.

        Returns:
            int: the size of the receptive field.

        """

        size = 1
        for n in range(self.num_layers):
            current_size = (self.kernel_size - 1) * (2 ** n)
            size += current_size
        return size
