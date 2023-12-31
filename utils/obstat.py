import numpy as np
from mpi4py import MPI


def sum_obstat(obstat1, obstat2, dtype):
    obstat1 += obstat2
    return obstat1


sumobstat_op = MPI.Op.Create(sum_obstat, commute=True)


class ObStat:
    def __init__(self, shape, eps=1e-2):
        self.eps = eps
        self.sum: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.sumsq: np.ndarray = np.full(shape, eps, dtype=np.float32)
        self.count: float = eps

    def inc(self, s: np.ndarray, ssq: np.ndarray, c: float):
        self.sum += s.astype(np.float32)
        self.sumsq += ssq.astype(np.float32)
        self.count += c

    def __repr__(self):
        return f'sum:{self.sum} sumsq:{self.sumsq} count:{self.count}'

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), self.eps))

    def mpi_inc(self, comm: MPI.Comm):
        stat = comm.allreduce(self, op=sumobstat_op)
        self.sum = stat.sum
        self.sumsq = stat.sumsq
        self.count = stat.count
