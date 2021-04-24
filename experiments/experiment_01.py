

import pytorch_lightning as pl
import optuna
import xarray as xr
import numpy as np

import os
import shutil
from argparse import ArgumentParser
from datetime import datetime

from project.fluxdata import FluxData
from models.hybrid import Q10Model


# All model and trainer args that do not change go here.
LOG_DIR = './logs/experiment_01/'
ARGS = [
    # '--fast_dev_run 1'
    '--limit_train_batches', '0.1',
    '--max_epochs', '20',
    '--log_every_n_steps', '1',
    f'--default_root_dir', LOG_DIR,
    '--learning_rate', '0.005',
    '--data_path', '/Net/Groups/BGI/people/bkraft/data/Synthetic4BookChap.nc',
    '--gpus', '1'
]

def objective(trial: optuna.trial.Trial) -> float:

    q10_init = trial.suggest_float('q10_init', 0.5, 2.5)
    seed = trial.suggest_int('seed', 0, 9)
    features = trial.suggest_categorical('features', ['sw_pot, dsw_pot', 'sw_pot, dsw_pot, ta'])
    features_parsed = features.split(', ')

    # ------------
    # args
    # ------------
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', default=240, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--data_path', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Q10Model.add_model_specific_args(parser)

    # Some default arguments (not best practice but cannot use `add_argument`).
    args = parser.parse_args(ARGS)

    if 'ta' in features_parsed:
        args.default_root_dir = os.path.join(LOG_DIR, 'w_ta')
    else:
        args.default_root_dir = os.path.join(LOG_DIR, 'n_ta')

    pl.seed_everything(seed)

    # These are the inputs to the NN.
    #features = ['sw_pot', 'dsw_pot']

    # Further variables used in the hybrid model.
    physical = ['ta']

    # Target (multiple targets not possible currently).
    targets = ['reco']

    # Find variables that are only needed in physical model but not in NN.
    physical_exclusive = [v for v in physical if v not in features_parsed]

    # ------------
    # data
    # ------------
    ds = xr.open_dataset(args.data_path)

    fluxdata = FluxData(
        ds,
        features=features_parsed + physical_exclusive,
        targets=targets,
        context_size=1,
        train_time=slice('2003-01-01', '2006-12-31'),
        valid_time=slice('2007-01-01', '2007-12-31'),
        test_time=slice('2008-01-01', '2008-12-31'),
        batch_size=args.batch_size,
        data_loader_kwargs={'num_workers': 4})

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
        features=features_parsed,
        targets=targets,
        norm=fluxdata._norm,
        ds=ds_pred,
        q10_init=q10_init,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, weights_summary=None)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    #trainer.test(test_dataloaders=test_loader)

    # ------------
    # save results
    # ------------
    # Store predictions.
    ds = fluxdata.add_scalar_record(model.ds, varname='q10', x=model.q10_history)

    # Add some attributes that are required for analysis.
    ds.attrs = {
        'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'author': 'bkraft@bgc-jena.mpg.de'
    }
    ds.q10.attrs = {'q10_init': args.q10_init, 'features': features}

    # Save data.
    save_dir = os.path.join(model.logger.log_dir, 'predictions.nc')
    print(f'Saving predictions to: {save_dir}')
    ds.to_netcdf(save_dir)

    return trainer.callback_metrics['valid_loss'].item()


def main():
    log_dir = LOG_DIR
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    search_space = {
        'q10_init': [0.5, 1.5, 2.5],
        'seed': [i for i in range(10)],
        'features': ['sw_pot, dsw_pot', 'sw_pot, dsw_pot, ta']
    }
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
    study.optimize(objective, n_jobs=1)

if __name__ == '__main__':
    main()
