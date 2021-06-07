
from joblib import Parallel, delayed
from argparse import ArgumentParser
import time

from experiments.experiment_01 import main as run_exp1

PYTHON_FILE = 'experiments/experiment_01.py'


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(self.name)
        elapsed = time.time() - self.tstart
        hours = elapsed // 3600
        elapsed = elapsed % 3600
        minutes = elapsed // 60
        elapsed = elapsed % 60
        seconds = elapsed
        print(f'>>> Elapsed: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s')


def main(parser: ArgumentParser = None, **kwargs):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument('--num_jobs', '-n', type=int, help='number of parallel workers', required=True)

    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    print('>>> Creating study')
    run_exp1(parser, create_study=True)

    print(f'>>> Running study with {args.num_jobs} workers')

    with Timer():
        Parallel(n_jobs=args.num_jobs)(delayed(run_exp1)(parser) for _ in range(args.num_jobs))


if __name__ == '__main__':
    main()
