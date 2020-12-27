
from pytorch_lightning import LightningModule, Trainer
from torch.optim import Optimizer, AdamW, SGD
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, \
    CyclicLR, OneCycleLR

from typing import Optional, Union


class GradualWarmupScheduler(LRScheduler):
    """Gradually warm-up learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Adapted from:

     Ildoo Kim / https://github.com/ildoonet/pytorch-gradual-warmup-lr

    Under license:

    ---------------------------------------------------------------------------
    MIT License

    Copyright (c) 2019 Ildoo Kim

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    ---------------------------------------------------------------------------
    """

    def __init__(
            self,
            optimizer: Optimizer,
            num_epochs: int,
            after_scheduler: Optional[LRScheduler] = None,
            multiplier: float = 1.0):
        """    Args:


        Args:
            optimizer (Optimizer): Wrapped optimizer.
            num_epochs (int): target learning rate is reached at num_epochs.
            after_scheduler (Optional[LRScheduler]): after `num_epochs`, use
                this scheduler (eg. ReduceLROnPlateau).
            multiplier (float): a learning rate multiplyer >= 1.0. Adjusts the
                target learning rate: lr = base_lr * multiplier if
                multiplier > 1.0. If multiplier = 1.0, lr starts from 0 and
                ends up with the base_lr. Defaults to 1.0.

        Raises:
            ValueError: multiplier must be >= 1.
        """

        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier must be >= 1.')
        self.num_epochs = num_epochs
        self.after_scheduler = after_scheduler
        self.finished = False

        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.num_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * ((float(self.last_epoch) + 1) /
                    (self.num_epochs + 1)) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch /
                    self.num_epochs + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        # ReduceLROnPlateau is called at the end of epoch, whereas others are
        # called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.num_epochs:
            warmup_lr = [
                base_lr * (
                    (self.multiplier - 1.) * self.last_epoch /
                    self.num_epochs + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.num_epochs)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.num_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# class LRStrategy(object):
#     """[summary]

#     Defines different learnign strategies (optimizer & (LR) scheduler). There
#     are many optimizers and schedulers, only a subset is implemented here.
#     In an effort to reducte the options, some scheduler arguments are
#     hard-coded.

#     ToDo: currently, AdamW foes only work with cyclic if cycle_momentum=False.

#     Optimizers:
#     ---------------------------------------------------------------------------

#     * `sgd`: stochastic gradient descent
#     * `adamw`: AdamW

#     SGD (sgd)
#     ---------

#     Standard stochastic gradient descent.

#     Args:
#         lr (float or list): Upper learning rate boundaries in the cycle
#             for each parameter group.
#         step_size (int): Number of training iterations in the
#             increasing half of a cycle.
#         mode (str): One of {`triangular`, `triangular2`}.
#             Values correspond to policies detailed below.
#             Defaults to `triangular`.

#     Schedulers:
#     ---------------------------------------------------------------------------

#     * `cosine`: CosineAnnealingWithWarmup
#     * `reduceonplateau`: ReduceLROnPlateauWithWarmup (currently not working)
#     * `cyclic`: CyclicLR scheduler
#     * `cyclic2`: CyclicLR scheduler
#     * `onecycle` OneCycleLR scheduler

#     CosineAnnealingWithWarmup (cosine)
#     ----------------------------------

#     After a warmup (linear increase to lr), the lr convergese to zero.

#     Args:
#         max_lr (float or list): Upper learning rate boundaries in the cycle
#             for each parameter group.
#         step_size (int): Number of training iterations in the
#             increasing half of a cycle.
#         mode (str): One of {`triangular`, `triangular2`}.
#             Values correspond to policies detailed below.
#             Defaults to `triangular`.


#          *  <-- lr
#         *      *
#        *         *
#       *             *       total_steps
#      *                  *     |
#     *                         * <-- base_lr

#     ReduceLROnPlateauWithWarmup (reduceonplateau)
#     ---------------------------------------------

#     *CURRENTLY NOT WORKING*

#     After a warmup (linear increase to lr), reduce learning rate when a metric
#     has stopped improving.

#     CyclicLR scheduler (cyclic & cyclic2)
#     -------------------------------------

#     ------ cyclic ------
#     Cyclic triangular with the option to reduce peaks with fixed decay.

#     Args:
#         max_lr (float or list): Upper learning rate boundaries in the cycle
#             for each parameter group.
#         step_size (int): Number of training iterations in the
#             increasing half of a cycle.
#         mode (str): One of {`triangular`, `triangular2`}.
#             Values correspond to policies detailed below.
#             Defaults to `triangular`.


#     Note: no warmup allowed with CLR scheduler.

#     mode = `triangular`:

#         step_size            cycle
#         |-------|       |---------------|
#                 * <-- lr        *               *
#               *   *           *   *           *   *
#             *       *       *       *       *       *
#           *           *   *           *   *           *
#         * <-- base_lr   *               *               *

#     ------ cyclic2 ------
#     mode = `triangular2` (max_lr halved each cycle):

#         step_size            cycle
#         |-------|       |---------------|
#                 * <-- lr
#               *   *
#             *       *           *
#           *           *     *       *           *
#         * <-- base_lr    *               *              *


#     OneCycleLR scheduler (onecycle)
#     -------------------------------

#     Only one cycle is performed (see shape below), for `super-convergence`.

#     Args:
#         max_lr (float or list): Upper learning rate boundaries in the cycle
#             for each parameter group.
#         total_steps (int): Number of training iterations.

#     Note: no warmup allowed with CLR scheduler.

#            total_steps
#     |------------------------|
#              *  <-- lr
#            *     *
#           *         *
#          *            *   total_steps
#        *                 *   |
#     * <-- base_lr            *

#     """

#     def __init__(
#             self,
#             lr: float,
#             weight_decay: float,
#             max_epochs: int,
#             batches_per_epoch: int,
#             optimizer: str,
#             scheduler: Optional[str] = None,
#             num_warmup_steps: Union[int, str] = 'auto',
#             num_cycles: int = 6,
#             base_lr: float = 1e-7,
#             optimizer_kwargs={},
#             scheduler_kwargs={}):

#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.base_lr = base_lr
#         self.max_epochs = max_epochs
#         self.batches_per_epoch = batches_per_epoch
#         self.max_steps = int(self.max_epochs * self.batches_per_epoch)

#         if isinstance(num_warmup_steps, str):
#             if num_warmup_steps != 'auto':
#                 raise ValueError(
#                     'argument `num_warmup_steps` must be an integer or '
#                     f'`auto`, is `{num_warmup_steps}`.'
#                 )
#             else:
#                 self.num_warmup_steps = self.batches_per_epoch
#         else:
#             self.num_warmup_steps = num_warmup_steps

#         self.num_cycles = num_cycles

#         self.half_cycle_step_size = self.max_steps / self.num_cycles // 2

#         self.optimizer = optimizer.lower()
#         self.scheduler = scheduler.lower()

#     def get_optimizer(self, model: LightningModule):
#         optimizer = self.optimizer

#         if optimizer == 'adamw':
#             optimizer = AdamW(
#                 model.parameters(),
#                 lr=self.lr,
#                 weight_decay=self.weight_decay)
#         elif optimizer == 'sgd':
#             optimizer = SGD(
#                 model.parameters(),
#                 lr=self.lr,
#                 weight_decay=self.weight_decay)
#         else:
#             raise ValueError(
#                 'attempt to create optimizer failed. The optimizer '
#                 f'`{optimizer}` is not a valid choice.'
#             )

#         return optimizer

#     def get_scheduler(self, optimizer: Optimizer):
#         scheduler = self.scheduler

#         if scheduler is None:
#             return None

#         if scheduler == 'cosine':
#             scheduler = CosineAnnealingLR(
#                 optimizer,
#                 T_max=self.max_steps - self.num_warmup_steps)
#             if self.num_warmup_steps > 0:
#                 scheduler = self.add_warmup(optimizer, scheduler)

#         # elif scheduler == 'reduceonplateau':
#         #     scheduler = ReduceLROnPlateau(optimizer)
#         #     if self.num_warmup_steps > 0:
#         #         scheduler = self.add_warmup(optimizer, scheduler)

#         elif scheduler == 'cyclic':
#             scheduler = CyclicLR(
#                 optimizer,
#                 base_lr=self.base_lr,
#                 max_lr=self.lr,
#                 step_size_up=self.half_cycle_step_size,
#                 mode='triangular',
#                 cycle_momentum=False if self.optimizer == 'adamw' else True)

#         elif scheduler == 'cyclic2':
#             scheduler = CyclicLR(
#                 optimizer,
#                 base_lr=self.base_lr,
#                 max_lr=self.lr,
#                 step_size_up=self.half_cycle_step_size,
#                 mode='triangular2',
#                 cycle_momentum=False if self.optimizer == 'adamw' else True)

#         elif scheduler == 'onecycle':
#             scheduler = OneCycleLR(
#                 optimizer,
#                 max_lr=self.lr,
#                 total_steps=self.max_steps)

#         else:
#             raise ValueError(
#                 'attempt to create scheduler failed. The scheduler '
#                 f'`{scheduler}` is not a valid choice.'
#             )

#         scheduler_dict = {
#             'scheduler': scheduler,
#             'interval': 'step'
#         }

#         return scheduler_dict

#     def get_optim_and_scheduler(self, model):
#         optimizer = self.get_optimizer(model)
#         scheduler = self.get_scheduler(optimizer)

#         return optimizer, scheduler

#     def add_warmup(self, optimizer, scheduler):
#         return GradualWarmupScheduler(
#             optimizer,
#             multiplier=1.0,
#             num_epochs=self.num_warmup_steps,
#             after_scheduler=scheduler
#         )

#     def simulate_lr(self):
#         """Simulates learning rate from 0:self.max_steps.

#         Returns:
#             List[float]: the learning rates.
#         """
#         import torch
#         model = torch.nn.Linear(1, 1)
#         optim = self.get_optimizer(model)
#         sched = self.get_scheduler(optim)['scheduler']

#         lr = []
#         for e in range(self.max_steps):
#             lr.append(sched.get_last_lr()[0])
#             sched.step()

#         return lr

class LRStrategy():
    """[summary]

    Defines different learnign strategies (optimizer & (LR) scheduler). There
    are many optimizers and schedulers, only a subset is implemented here.
    In an effort to reducte the options, some scheduler arguments are
    hard-coded.

    ToDo: currently, AdamW foes only work with cyclic if cycle_momentum=False.

    Optimizers:
    ---------------------------------------------------------------------------

    * `sgd`: stochastic gradient descent
    * `adamw`: AdamW

    SGD (sgd)
    ---------

    Standard stochastic gradient descent.

    Args:
        lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        step_size (int): Number of training iterations in the
            increasing half of a cycle.
        mode (str): One of {`triangular`, `triangular2`}.
            Values correspond to policies detailed below.
            Defaults to `triangular`.

    Schedulers:
    ---------------------------------------------------------------------------

    * `cosine`: CosineAnnealingWithWarmup
    * `reduceonplateau`: ReduceLROnPlateauWithWarmup (currently not working)
    * `cyclic`: CyclicLR scheduler
    * `cyclic2`: CyclicLR scheduler
    * `onecycle` OneCycleLR scheduler

    CosineAnnealingWithWarmup (cosine)
    ----------------------------------

    After a warmup (linear increase to lr), the lr convergese to zero.

    Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        step_size (int): Number of training iterations in the
            increasing half of a cycle.
        mode (str): One of {`triangular`, `triangular2`}.
            Values correspond to policies detailed below.
            Defaults to `triangular`.


         *  <-- lr
        *      *
       *         *
      *             *       total_steps
     *                  *     |
    *                         * <-- base_lr

    ReduceLROnPlateauWithWarmup (reduceonplateau)
    ---------------------------------------------

    *CURRENTLY NOT WORKING*

    After a warmup (linear increase to lr), reduce learning rate when a metric
    has stopped improving.

    CyclicLR scheduler (cyclic & cyclic2)
    -------------------------------------

    ------ cyclic ------
    Cyclic triangular with the option to reduce peaks with fixed decay.

    Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        step_size (int): Number of training iterations in the
            increasing half of a cycle.
        mode (str): One of {`triangular`, `triangular2`}.
            Values correspond to policies detailed below.
            Defaults to `triangular`.


    Note: no warmup allowed with CLR scheduler.

    mode = `triangular`:

        step_size            cycle
        |-------|       |---------------|
                * <-- lr        *               *
              *   *           *   *           *   *
            *       *       *       *       *       *
          *           *   *           *   *           *
        * <-- base_lr   *               *               *

    ------ cyclic2 ------
    mode = `triangular2` (max_lr halved each cycle):

        step_size            cycle
        |-------|       |---------------|
                * <-- lr
              *   *
            *       *           *
          *           *     *       *           *
        * <-- base_lr    *               *              *


    OneCycleLR scheduler (onecycle)
    -------------------------------

    Only one cycle is performed (see shape below), for `super-convergence`.

    Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): Number of training iterations.

    Note: no warmup allowed with CLR scheduler.

           total_steps
    |------------------------|
             *  <-- lr
           *     *
          *         *
         *            *   total_steps
       *                 *   |
    * <-- base_lr            *

    """

    def __init__(
            self,
            lr: float,
            weight_decay: float,
            optimizer: str,
            scheduler: Optional[str] = None,
            num_warmup_steps: Union[int, str] = 'auto',
            num_cycles: int = 6,
            base_lr: float = 1e-7):
        """Initialize learning strategy.

        Args:
            lr (float): the learning rate (>0).
            weight_decay (float): the weight decay (>=0).
            optimizer (str): an optimizer name, one of {`adamw`, `sgd`}.
            scheduler (Optional[str], optional): the learning rate scheduler,
                one of {`cosine`, `reduceonplateau`, `cyclic`, `cyclic2`,
                `onecycle`}. Defaults to None.
            num_warmup_steps (Union[int, str], optional): the number of warmup
                steps. Does not apply to all schedulers (cyclic and onecycle
                do start at low lr anyway). No warmup is done if `0`, one full
                epoch (gradually increasing per batch) if `auto`. Defaults to
                `auto`.
            num_cycles (int, optional): the number of cycles if a cyclic lr
                scheduler is chosen. Defaults to 6.
            base_lr (float, optional): the base learning rate used by some
                schedulers. This is the minimum learning rate. Defaults
                to 1e-7.
        """

        self.lr = lr
        self.weight_decay = weight_decay
        self.base_lr = base_lr

        self.optimizer = optimizer.lower()
        self.scheduler = scheduler.lower()

        self.num_warmup_steps = num_warmup_steps
        self.num_cycles = num_cycles
        self.base_lr = base_lr

    def get_optimizer(self, model: LightningModule):
        optimizer = self.optimizer

        if optimizer == 'adamw':
            optimizer = AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)
        elif optimizer == 'sgd':
            optimizer = SGD(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)
        else:
            raise ValueError(
                'attempt to create optimizer failed. The optimizer '
                f'`{optimizer}` is not a valid choice.'
            )

        return optimizer

    def get_num_warmup(self, batches_per_epoch):
        if isinstance(self.num_warmup_steps, str) and self.num_warmup_steps == 'auto':
            return batches_per_epoch
        elif isinstance(self.num_warmup_steps, int):
            return self.num_warmup_steps
        else:
            raise ValueError(
                'argument `num_warmup_steps` must be an integer or '
                f'`auto`, is `{self.num_warmup_steps}`.'
            )

    def get_scheduler(
            self,
            optimizer: Optimizer,
            max_epochs: int,
            batches_per_epoch: int):

        if self.scheduler is None:
            return None

        scheduler = self.scheduler

        max_steps = int(max_epochs * batches_per_epoch)

        half_cycle_size = max_steps / self.num_cycles // 2
        num_warmup_steps = self.get_num_warmup(batches_per_epoch)

        if scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_steps - num_warmup_steps)
            if num_warmup_steps > 0:
                scheduler = self.add_warmup(optimizer, scheduler, num_warmup_steps)

        # elif scheduler == 'reduceonplateau':
        #     scheduler = ReduceLROnPlateau(optimizer)
        #     if self.num_warmup_steps > 0:
        #         scheduler = self.add_warmup(optimizer, scheduler)

        elif scheduler == 'cyclic':
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.base_lr,
                max_lr=self.lr,
                step_size_up=half_cycle_size,
                mode='triangular',
                cycle_momentum=False if self.optimizer == 'adamw' else True)

        elif scheduler == 'cyclic2':
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.base_lr,
                max_lr=self.lr,
                step_size_up=half_cycle_size,
                mode='triangular2',
                cycle_momentum=False if self.optimizer == 'adamw' else True)

        elif scheduler == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=max_steps)

        else:
            raise ValueError(
                'attempt to create scheduler failed. The scheduler '
                f'`{scheduler}` is not a valid choice.'
            )

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': f'{self.scheduler}_lr',
            'frequency': 1
        }

        return scheduler_dict

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):

        for lr_scheduler in trainer.lr_schedulers:
            if hasattr(lr_scheduler['scheduler'], '__call__'):
                scheduler = lr_scheduler['scheduler']

                n_train = len(pl_module.train_dataloader())
                n_accumulate_grad = trainer.accumulate_grad_batches
                n_max_epochs = trainer.max_epochs

                # or trainer.tpu_cores if tpu or 1 if cpu
                n_devices = trainer.num_gpus

                num_training_steps = n_train // n_accumulate_grad * \
                    n_max_epochs // n_devices

                lr_scheduler['scheduler'] = scheduler(
                    max_steps=num_training_steps,
                    batches_per_epoch=n_train)

    def add_warmup(self, optimizer: Optimizer, scheduler: LRScheduler, num_warmup_steps: int):
        return GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            num_epochs=num_warmup_steps,
            after_scheduler=scheduler
        )

    def simulate_lr(self):
        """Simulates learning rate from 0:self.max_steps.

        Returns:
            List[float]: the learning rates.
        """
        import torch
        model = torch.nn.Linear(1, 1)
        optim = self.get_optimizer(model)
        sched = self.get_scheduler(optim)['scheduler']

        lr = []
        for e in range(self.max_steps):
            lr.append(sched.get_last_lr()[0])
            sched.step()

        return lr
