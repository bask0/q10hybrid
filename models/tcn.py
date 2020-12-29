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

from torch import Tensor
import torch.nn as nn
from torch.nn.utils import weight_norm

from typing import Dict

from utils.pl_utils import LightningNet
from utils.torch_utils import Transform


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class Residual(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        """Residual connection, does downsampling if necessary.

        Args:
            n_inputs (int): layer input size (number of channels).
            n_outputs (int): layer output size (number of channels).
        """
        super(Residual, self).__init__()
        self.do_downsample = n_inputs != n_outputs
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if self.do_downsample else nn.Identity()

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        return x + self.downsample(res)

    def init_weights(self) -> None:
        if self.do_downsample:
            self.downsample.weight.data.normal_(0, 0.01)


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
        """Implements a two-layered residual block.

        Args:
            n_inputs (int): number of inputs (channels).
            n_outputs (int): number of outputs (channels).
            kernel_size (int): the 1D convolution kernel size.
            stride (int): the 1D convolution stride.
            dilation (int): the 1D convolution dilation.
            padding (int): the padding.
            dropout (float, optional): the dropout applied after each layer. Defaults to 0.2.
        """

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

        self.res = Residual(n_inputs, n_outputs)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.res.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Model forward run.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.res(out, x)
        return self.relu(out)


class TemporalConvNet(LightningNet):
    def __init__(
            self,
            training_config: Dict,
            num_inputs: int,
            num_outputs: int,
            num_hidden: int,
            kernel_size: int = 4,
            num_layers: int = 2,
            dropout: float = 0.0) -> None:
        """Implements a Temporal Convolutional Network (TCN).

        https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

        Note:
            The TCN layer is followed by a feedfoward layer to map the TCN output channels to `num_outputs`.

        Shapes:
            Input:  (batch_size, input_size, sequence_length)
            Output: (batch_size, num_channels[-1], sequence_length)

        Args:
            training_config (Dict): the training configuration passed to the superclass `LightningNet`. This is at
                least: `lr`, `weight_decay`, `max_epochs`.
            num_inputs (int): the mumber of input features.
            num_intputs (int): the number of outputs.
            num_hidden (int): the hidden size (intermediate channel sizes) of the layers.
            kernel_size (int): the kernel size. Defaults to 4.
            num_layers (int): the number of stacked layers. Defaults to 2.
            dropout (float): a float value in the range [0, 1) that defines the dropout probability. Defaults to 0.0.
        """

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

        self.tcn = nn.Sequential(*layers)

        self.to_channel_last = Transform(transform_fun=lambda x: x.permute(0, 2, 1))

        self.linear = nn.Linear(num_hidden, num_outputs)

        self.to_sequence_last = Transform(transform_fun=lambda x: x.permute(0, 2, 1))

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        """Run data through the model.

        Args:
            x (Tensor): the input data with shape (batch, num_inputs, seq).

        Returns:
            Tensor: the model output with shape (batch, seq, num_outputs).
        """
        out = self.tcn(x)
        out = self.to_channel_last(out)
        out = self.linear(out)
        out = self.to_sequence_last(out)
        return out

    def receptive_field_size(self) -> int:
        """Returns the receptive field of the Module.

        The receptive field (number of steps the model looks back) of the model depends
        on the number of layers and the kernel size.

        Returns:
            int: the size of the receptive field.

        """

        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)
