from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict

import numpy as np
# from mlflow import log_params, log_metric, log_metrics, set_experiment, start_run
from mpi4py import MPI

from evo.training_result import TrainingResult


class Reporter(ABC):
    @abstractmethod
    def start_gen(self):
        pass

    @abstractmethod
    def log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, time: float):
        pass

    @abstractmethod
    def end_gen(self):
        pass

    @abstractmethod
    def print(self, s: str):
        """For printing one time information"""
        pass

    @abstractmethod
    def log(self, d: Dict[str, float]):
        """For logging key value pairs that recur each generation"""
        pass


class ReporterSet(Reporter):
    def __init__(self, *reporters: Reporter):
        self.reporters = [reporter for reporter in reporters if reporter is not None]

    def start_gen(self):
        for reporter in self.reporters:
            reporter.start_gen()

    def log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, time: float):
        for reporter in self.reporters:
            reporter.log_gen(fits, noiseless_tr, time)

    def end_gen(self):
        for reporter in self.reporters:
            reporter.end_gen()

    def print(self, s: str):
        for reporter in self.reporters:
            reporter.print(s)

    def log(self, d: Dict[str, float]):
        for reporter in self.reporters:
            reporter.log(d)


class MPIReporter(Reporter, ABC):
    MAIN = 0

    def __init__(self, comm: MPI.Comm):
        self.comm = comm

    def start_gen(self):
        if self.comm.rank == MPIReporter.MAIN:
            self._start_gen()

    def log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, time: float):
        if self.comm.rank == MPIReporter.MAIN:
            self._log_gen(fits, noiseless_tr, time)

    def end_gen(self):
        if self.comm.rank == MPIReporter.MAIN:
            self._end_gen()

    def print(self, s: str):
        if self.comm.rank == MPIReporter.MAIN:
            self._print(s)

    def log(self, d: Dict[str, float]):
        if self.comm.rank == MPIReporter.MAIN:
            self._log(d)

    @abstractmethod
    def _start_gen(self):
        pass

    @abstractmethod
    def _log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, time: float):
        pass

    @abstractmethod
    def _end_gen(self):
        pass

    @abstractmethod
    def _print(self, s: str):
        pass

    @abstractmethod
    def _log(self, d: Dict[str, float]):
        pass


class StdoutReporter(MPIReporter):
    def __init__(self, comm: MPI.Comm):
        super().__init__(comm)
        if comm.rank == 0:
            self.gen = 0
            # self.cum_steps = 0

    def _start_gen(self):
        print(f'\n\n'
              f'----------------------------------------'
              f'\ngen:{self.gen}')

    def _log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, time: float):
        for i, col in enumerate(fits.T):
            # Objectives are grouped by column so this finds the avg and max of each objective
            print(f'obj {i} avg:{np.mean(col):0.2f}')
            print(f'obj {i} max:{np.max(col):0.2f}')

        print(f'fit:{noiseless_tr.result[0]}')

        print(f'dist:{noiseless_tr.dist}')
        print(f'rew:{noiseless_tr.reward}')

        print(f'time:{time:0.2f}')

    def _end_gen(self):
        self.gen += 1

    def _print(self, s: str):
        print(s)

    def _log(self, d: Dict[str, float]):
        for k, v in d.items():
            print(f'{k}:{v}')


class LoggerReporter(MPIReporter):
    def __init__(self, comm: MPI.Comm, cfg, log_path=None):
        super().__init__(comm)

        if comm.rank == 0:
            if log_path is None:
                log_path = '/logs'
            self.log_name = f'mg__' + datetime.now().strftime('%d_%m_%y-%H_%M_%S')

            logging.basicConfig(filename=f'{log_path}/{self.log_name}.log', level=logging.DEBUG, force=True)
            logging.info('initialized logger')

            self.gen = 0
            self.cfg = cfg

            self.best_rew = 0
            self.best_dist = 0
            # self.cum_steps = 0

    def _start_gen(self):
        logging.info(f'gen:{self.gen}')

    def _log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, time: float):
        for i, col in enumerate(fits.T):
            # Objectives are grouped by column so this finds the avg and max of each objective
            logging.info(f'obj {i} avg:{np.mean(col):0.2f}')
            logging.info(f'obj {i} max:{np.max(col):0.2f}')

        logging.info(f'fit:{noiseless_tr.result[0]}')

        logging.info(f'dist:{noiseless_tr.dist}')
        logging.info(f'rew:{noiseless_tr.reward}')

        logging.info(f'time:{time:0.2f}')

    def _end_gen(self):
        self.gen += 1

    def _print(self, s: str):
        logging.info(s)

    def _log(self, d: Dict[str, float]):
        for k, v in d.items():
            logging.info(f'{k}:{v}')
