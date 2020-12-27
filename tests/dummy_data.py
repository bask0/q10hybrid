"""Dummy data, randomly generated."""

from torch.utils import data
from pytorch_lightning import LightningDataModule
import numpy as np


class DummyDataSimple(data.Dataset):
    def __init__(self, num_epochs: int = 1000, error_scale: float = 0.1):
        """This is a dummy dataset.

        Returns a x, y tuple:
            y = 0.5 + 1.5 x + e
            where x ~ N(0, 1), e ~ N(0, 0.1)

        Args:
            num_epochs (int): the epoch size.
            error_scale (float): the random normal error standard deviation. Defaults to 0.1.
        """

        self.num_epochs = num_epochs
        self.error_scale = error_scale

    def __len__(self):
        """The dataset length (number of samples), required.

        Returns:
            int: the length.
        """
        return 1000

    def __getitem__(self, idx: int):
        """Returns a single item corresponding to the index `ind`.

        Args:
            idx (int): the index of the sample, in range [0, len(self)).

        Returns:
            Tuple: a tuple of tensors of shape (1,) each.
        """
        x = np.random.normal(size=1).astype(np.float32)
        y = 0.5 + 1.5 * x + np.random.normal(scale=self.error_scale, size=(1,)).astype(np.float32)

        return x, y


class DummyDataSequential(data.Dataset):
    def __init__(
            self,
            num_epochs: int = 1000,
            seq_length: int = 20,
            error_scale: float = 0.1,
            seq_last: bool = False):
        """This is a dummy dataset.

        Returns a x, y tuple:
            y_{t} = x_{t-1} + e
            where x ~ N(0, 1), e ~ N(0, 0.1)

        y is x shifted by one, plus a linear transform.

        An epoch has 1'000 samples.

        Attrs:
            num_epochs (int): the epoch size. Defaults to 1'000.
            seq_length (int): the length of the simulated sequence. Defaults to 20.
            error_scale (float): the random normal error standard deviation. Defaults to 0.1.
            seq_last (bool): If `False`, the sequence dimension is the second last dimension, and the feature dimension
                is the first, else reversed. Defaults to `False`.
        """

        self.num_epochs = num_epochs
        self.seq_length = seq_length
        self.error_scale = error_scale
        self.seq_last = seq_last

    def __len__(self):
        """The dataset length (number of samples), required.

        Returns:
            int: the length.
        """
        return self.num_epochs

    def __getitem__(self, idx: int):
        """Returns a single item corresponding to the index `ind`.

        The target `y_[t]` is linear transformation of `x_[t-1]`.

        Args:
            idx (int): the index of the sample, in range [0, len(self)).

        Returns:
            Tuple: A tuple of tensors (x, y) of shape (seq_length, 1) each.
        """
        x = np.random.normal(size=(self.seq_length + 1, 1)).astype(np.float32)
        y = 0.5 + 1.5 * x[:-1, :] + \
            np.random.normal(scale=self.error_scale, size=(self.seq_length, 1)).astype(np.float32)

        x = x[1:]

        if self.seq_last:
            x = x.T
            y = y.T

        return x, y


class DataSimple(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return data.DataLoader(DummyDataSimple(), batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(DummyDataSimple(), batch_size=self.batch_size)

    def test_dataloader(self):
        return data.DataLoader(DummyDataSimple(), batch_size=self.batch_size)


class DataSequential(LightningDataModule):
    def __init__(self, batch_size, seq_last):
        super().__init__()
        self.batch_size = batch_size
        self.seq_last = seq_last

    def train_dataloader(self):
        return data.DataLoader(DummyDataSequential(), batch_size=self.batch_size, seq_last=self.seq_last)

    def val_dataloader(self):
        return data.DataLoader(DummyDataSequential(), batch_size=self.batch_size, seq_last=self.seq_last)

    def test_dataloader(self):
        return data.DataLoader(DummyDataSequential(), batch_size=self.batch_size, seq_last=self.seq_last)
