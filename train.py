
import xarray as xr
from argparse import ArgumentParser
import pytorch_lightning as pl
import os

from project.fluxdata import FluxData
from models.hybrid import Q10Model


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', default=160, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Q10Model.add_model_specific_args(parser)

    # Some default arguments (not best practice but cannot use `add_argument`).
    args = parser.parse_args(
        [
            '--limit_train_batches', '0.1', # Small training set size because we monitor predictions.
            '--max_epochs', '15',
            '--log_every_n_steps', '1'
        ]
    )

    # These are the inputs to the NN.
    features = ['sw_pot', 'dsw_pot']

    # Further variables used in the hybrid model.
    physical = ['ta']

    # Target (multiple targets not possible currently).
    targets = ['reco']

    # Find variables that are only needed in physical model but not in NN.
    physical_exclusive = [v for v in physical if v not in features]

    # ------------
    # data
    # ------------
    ds = xr.open_dataset('/Users/bk/Files/usmile/data/q10/Synthetic4BookChap.nc')

    fluxdata = FluxData(
        ds,
        features=features + physical_exclusive,
        targets=targets,
        context_size=1,
        train_time=slice('2003-01-01', '2006-12-31'),
        valid_time=slice('2007-01-01', '2007-12-31'),
        test_time=slice('2008-01-01', '2008-12-31'),
        batch_size=args.batch_size,
        data_loader_kwargs={'num_workers': 2})

    train_loader = fluxdata.train_dataloader()
    val_loader = fluxdata.val_dataloader()
    test_loader = fluxdata.test_dataloader()

    # Create empty xr.Dataset, will be used by the model to save predictions every epoch.
    max_epochs = args.max_epochs if args.max_epochs is not None else 500
    ds_pred = fluxdata.target_xr('valid', varnames=['reco', 'rb'], num_epochs=max_epochs)

    # ------------
    # model
    # ------------
    model = Q10Model(
        features=features,
        targets=targets,
        norm=fluxdata._norm,
        ds=ds_pred,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)

    # ------------
    # save results
    # ------------
    # Save data.
    ds = fluxdata.add_scalar_record(model.ds, varname='q10', x=model.q10_history)
    save_dir = os.path.join(model.logger.log_dir, 'predictions.nc')
    print(f'Saving predictions to: {save_dir}')
    ds.to_netcdf(save_dir)


if __name__ == '__main__':
    cli_main()
