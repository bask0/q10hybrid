

import pytorch_lightning as pl
import optuna
import xarray as xr

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import os
import shutil
from argparse import ArgumentParser
from datetime import datetime

from project.fluxdata import FluxData
from models.hybrid import Q10Model

# Hardcoded `Trainer` args. Note that these cannot be passed via cli.
TRAINER_ARGS = dict(
    max_epochs=10,
    log_every_n_steps=1,
    gpus=1,
    weights_summary=None
)


class Objective(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, trial: optuna.trial.Trial) -> float:
        q10_init = trial.suggest_float('q10_init', 0.0001, 1000.)
        seed = trial.suggest_int('seed', 0, 999999999999)
        weight_decay = trial.suggest_float('weight_decay', 0., 1000.)
        use_ta  = trial.suggest_categorical('use_ta', [True, False])
        dropout = trial.suggest_float('dropout', 0.0, 1.0)

        if use_ta:
            features = ['sw_pot', 'dsw_pot', 'ta']
        else:
            features = ['sw_pot', 'dsw_pot']

        pl.seed_everything(seed)

        # These are the inputs to the NN.
        #features = ['sw_pot', 'dsw_pot']

        # Further variables used in the hybrid model.
        physical = ['ta']

        # Target (multiple targets not possible currently).
        targets = ['reco']

        # Find variables that are only needed in physical model but not in NN.
        physical_exclusive = [v for v in physical if v not in features]

        # ------------
        # data
        # ------------
        ds = xr.open_dataset(self.args.data_path)

        fluxdata = FluxData(
            ds,
            features=features + physical_exclusive,
            targets=targets,
            context_size=1,
            train_time=slice('2003-01-01', '2006-12-31'),
            valid_time=slice('2007-01-01', '2007-12-31'),
            test_time=slice('2008-01-01', '2008-12-31'),
            batch_size=self.args.batch_size,
            data_loader_kwargs={'num_workers': 4})

        train_loader = fluxdata.train_dataloader()
        val_loader = fluxdata.val_dataloader()
        test_loader = fluxdata.test_dataloader()

        # Create empty xr.Dataset, will be used by the model to save predictions every epoch.
        max_epochs = TRAINER_ARGS['max_epochs']
        ds_pred = fluxdata.target_xr('valid', varnames=['reco', 'rb'], num_epochs=max_epochs)

        # ------------
        # model
        # ------------
        model = Q10Model(
            features=features,
            targets=targets,
            norm=fluxdata._norm,
            ds=ds_pred,
            q10_init=q10_init,
            hidden_dim=self.args.hidden_dim,
            num_layers=self.args.num_layers,
            learning_rate=self.args.learning_rate,
            dropout=dropout,
            weight_decay=weight_decay,
            num_steps=len(train_loader) * max_epochs)

        # ------------
        # training
        # ------------
        trainer = pl.Trainer.from_argparse_args(
            self.args,
            default_root_dir=self.args.log_dir,
            **TRAINER_ARGS,
             callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    min_delta=0.001),
                ModelCheckpoint(
                    filename='{epoch}-{val_loss:.2f}',
                    save_top_k=1,
                    verbose=False,
                    monitor='val_loss',
                    mode='min',
                    prefix=model.__class__.__name__)
        ])
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
        trial.set_user_attr('q10', ds.q10[-1].item())

        # Add some attributes that are required for analysis.
        ds.attrs = {
            'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'author': 'bkraft@bgc-jena.mpg.de',
            'q10_init': q10_init,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'use_ta': int(use_ta),
            'loss': trainer.callback_metrics['valid_loss'].item()
        }

        ds = ds.isel(epoch=slice(0, trainer.current_epoch + 1))

        # Save data.
        save_dir = os.path.join(model.logger.log_dir, 'predictions.nc')
        print(f'Saving predictions to: {save_dir}')
        ds.to_netcdf(save_dir)

        return trainer.callback_metrics['valid_loss'].item()

    @staticmethod
    def add_project_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--restart', action='store_true')
        parser.add_argument(
            '--batch_size', default=240, type=int)
        parser.add_argument(
            '--lr_cycle_size', default=40, type=int)
        parser.add_argument(
            '--data_path', default='/Net/Groups/BGI/people/bkraft/data/Synthetic4BookChap.nc', type=str)
        parser.add_argument(
            '--log_dir', default='./logs/experiment_01/', type=str)
        return parser


def main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = Objective.add_project_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Q10Model.add_model_specific_args(parser)
    parser.add_argument('--new_study', action='store_true', help='create new study (deletes old) and exits')
    parser.add_argument('--single_seed', action='store_true', help='use only one seed instead of (1, ..., 10).')
    args = parser.parse_args()

    # ------------
    # study setup
    # ------------
    search_space = {
        'q10_init': [0.5, 1.5, 2.5],
        'seed': [0] if args.single_seed else [i for i in range(10)],
        'weight_decay': [0.0, 0.01, 0.1],
        'dropout': [0.0, 0.2, 0.4],
        'use_ta': [True, False]
    }

    sql_file = os.path.abspath(os.path.join(args.log_dir, "optuna.db"))
    sql_path = f'sqlite:///{sql_file}'

    if args.new_study | args.restart | (not os.path.isfile(sql_file)):
        shutil.rmtree(args.log_dir, ignore_errors=True)
        os.makedirs(args.log_dir)
        study = optuna.create_study(
            study_name="q10hybrid",
            storage=sql_path,
            sampler=optuna.samplers.GridSampler(search_space),
            direction='minimize',
            load_if_exists=False)

        if args.new_study:
            exit()

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # ------------
    # run study
    # ------------
    n_trials = 1
    for k, v in search_space.items():
        n_trials *= len(v)
    study = optuna.load_study(
        study_name="q10hybrid",
        storage=sql_path,
        sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(Objective(args), n_trials=n_trials)

if __name__ == '__main__':
    main()
