"""Fluxnet data loaders."""

from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from pytorch_lightning import LightningDataModule

import xarray as xr
import numpy as np

from typing import List, Tuple, Dict, Union, Iterable

from utils.data_utils import Normalize


class FDataset(Dataset):
    """Fluxnet site data.

    Args:
        ds (xr.Dataset):
            Site cube data with dimension `time`.
        features (str):
            List of feature names.
        targets (str):
            List of target names.
        context_size (int):
            Context length (t-context_size+1 : t).
        norm (Normalize):
            Normalization module with all features and targets registered.
    """

    def __init__(
            self,
            ds: xr.Dataset,
            features: List[str],
            targets: List[str],
            context_size: int,
            norm: Normalize) -> None:

        self._ds = ds
        self._features = [features] if isinstance(features, str) else features
        self._targets = [targets] if isinstance(targets, str) else targets
        self._variables = self._features + self._targets
        self._context_size = context_size
        self._norm = norm

        self._ind2coord_lookup = np.argwhere(
            self._ds[self._targets].notnull().to_array().any('variable').values).flatten()

    def __len__(self) -> int:
        """The dataset length (number of samples), required.

        Returns:
            int: the length.
        """
        return len(self._ind2coord_lookup)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Returns a single item corresponding to the index `ind`.

        Args:
            idx (int):
                The index of the sample, in range [0, len(self)).

        Returns:
            Tuple:
                - Data (Dict[str, torch.Tensor]): features and targets, each with shape (1, context_size)
                - Time index (torch.Tensor): the time index corresponding to the data.
        """

        time = self._ind2coord_lookup[idx]

        ds_site_x = self._ds[self._variables].isel(time=slice(time - self._context_size + 1, time + 1))

        return {var: ds_site_x[var].values.astype('float32') for var in self._variables}, idx

    @property
    def num_features(self) -> int:
        return len(self._features)

    @property
    def num_targets(self) -> int:
        return len(self._targets)

    @property
    def num_variables(self) -> int:
        return self.num_features + self.num_targets


class FluxData(LightningDataModule):
    """Fluxnet site data.

    Args:
        ds (xr.Dataset):
            Site cube data with dimension `time`.
        features (str):
            List of feature names.
        targets (str):
            List of target names.
        context_size (int):
            Context length (t-context_size+1 : t).
        train_time (slice):
            Slice containg start and end time, e.g., slice('2001-01-01', '2001-12-31') or the
            training data.
        valid_time (slice):
            Slice containg start and end time, e.g., slice('2001-01-01', '2001-12-31') or the
            validation data.
        test_time (slice):
            Slice containg start and end time, e.g., slice('2001-01-01', '2001-12-31') or the
            test data.
        batch_size (int):
            The batch size.
        data_loader_kwargs (Dict, optional):
            Keyword arguments passed to Dataloader when calling one of the
            `[set]_dataloader` methods. Defaults is passing no further arguments.
    """
    def __init__(
            self,
            ds: xr.Dataset,
            features: List[str],
            targets: List[str],
            train_time: slice,
            valid_time: slice,
            test_time: slice,
            context_size: int,
            batch_size: int,
            data_loader_kwargs: Dict = {}) -> None:

        super().__init__()

        self._ds = ds
        self._features = [features] if isinstance(features, str) else features
        self._targets = [targets] if isinstance(targets, str) else targets
        self._train_time = train_time
        self._valid_time = valid_time
        self._test_time = test_time
        self._context_size = context_size
        self._batch_size = batch_size
        self._data_loader_kwargs = data_loader_kwargs

        self._ds_train = self._ds.sel(time=self._train_time).load()
        self._ds_valid = self._ds.sel(time=self._valid_time).load()
        self._ds_test = self._ds.sel(time=self._test_time).load()

        # Register normalization parameters from training data.
        self._norm = Normalize()
        self._norm.register_xr(self._ds_train, self._features + self._targets)

        # These are constant kwargs to FDataset.
        self._datakwargs = {
            'features': self._features,
            'targets': self._targets,
            'context_size': self._context_size,
            'norm': self._norm
        }

    @property
    def features(self) -> List[str]:
        return self._features

    @property
    def targets(self) -> List[str]:
        return self._targets

    @property
    def num_features(self) -> int:
        return len(self._features)

    @property
    def num_targets(self) -> int:
        return len(self._targets)

    def train_dataloader(self) -> DataLoader:
        """"Get the training dataloader."""
        return DataLoader(
            FDataset(
                self._ds_train,
                **self._datakwargs),
            batch_size=self._batch_size,
            shuffle=True,
            **self._data_loader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        """"Get the validation dataloader."""
        return DataLoader(
            FDataset(
                self._ds_valid,
                **self._datakwargs),
            batch_size=self._batch_size,
            shuffle=False,
            **self._data_loader_kwargs
        )

    def test_dataloader(self) -> DataLoader:
        """"Get the testing dataloader."""
        return DataLoader(
            FDataset(
                self._ds_test,
                **self._datakwargs),
            batch_size=self._batch_size,
            shuffle=False,
            **self._data_loader_kwargs
        )

    def target_xr(
            self,
            mode: str,
            varnames: Union[str, List[str]],
            num_epochs: int = 1) -> xr.Dataset:
        if mode not in ('train', 'valid', 'test'):
            raise ValueError(
                f'`mode` must be on of (`train` | `valid` | `test`), is `{mode}`.'
            )

        if mode == 'train':
            ds = self._ds_train
        elif mode == 'valid':
            ds = self._ds_valid
        elif mode == 'test':
            ds = self._ds_test
        else:
            raise ValueError(
                f'`mode` must be on of (`train` | `valid` | `test`), is `{mode}`.'
            )

        varnames = [varnames] if isinstance(varnames, str) else varnames

        ds_new = ds[varnames]

        for var in varnames:
            var_new = var + '_pred'
            dummy = ds[var].copy()
            dummy.values[:] = np.nan
            dummy = dummy.expand_dims(epoch=np.arange(num_epochs))
            ds_new[var_new] = dummy.copy()

        return ds_new

    def add_scalar_record(self, ds: xr.Dataset, varname: str, x: Iterable) -> xr.Dataset:

        if isinstance(x, Tensor):
            x = x.detach().cpu().numpy()

        # Cut excess entries (NaNs).
        x = x[:x.argmin()]

        if 'iter' not in ds.coords:
            ds = ds.assign_coords({'iter': np.arange(len(x))})
        else:
            if len(ds['iter']) != len(x):
                raise ValueError(
                    f'dimension `iter` already exists in `ds`, but length ({len(ds["iter"])}) does '
                    f'not match length of `x` ({len(x)}).'
                )

        ds[varname] = ('iter', x)

        return ds

    def teardown(self) -> None:
        """Clean up after fit or test, called on every process in DDP."""
        self._ds_train.close()
        self._ds_valid.close()
        self._ds_test.close()
