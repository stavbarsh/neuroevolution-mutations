import argparse
import json
from types import SimpleNamespace
from typing import List
import random
import numpy as np
import torch
from mpi4py import MPI


def parse_args():
    parser = argparse.ArgumentParser(description='maze-generality')
    parser.add_argument('config', type=str, help='Config file that will be used')
    parser.add_argument('--force_env', type=str, help='Override environment', default=None)
    parser.add_argument('--force_seed', type=int, help='Override seed', default=None)
    parser.add_argument('--force_obnorm', type=int, help='Override ob_norm', default=None)
    parser.add_argument('--force_lr', type=float, help='Override lr', default=None)
    parser.add_argument('--force_gens', type=int, help='Override gens', default=None)
    return parser.parse_args()


def load_config(cfg_file: str):
    """:returns: a SimpleNamespace from a json file"""
    return json.load(open(cfg_file), object_hook=lambda d: SimpleNamespace(**d))


def generate_seed(comm: MPI.Comm) -> int:
    return comm.scatter([np.random.randint(0, 1000000)] * comm.size)


def seed(comm: MPI.Comm, sd: int) -> np.random.RandomState:
    """Seeds torch and returns a random state that is different for each MPI proc"""
    print(f'comm: {comm.rank}, rs seed: {sd + 10000 * comm.rank}')
    rs = np.random.RandomState(sd + 10000 * comm.rank)  # This seed must be different on each proc
    random.seed(sd)  # this is seeded just in case since only np.random is used
    torch.random.manual_seed(0)  # This seed must be the same on each proc for generating initial params
    # also it should be the same between different experiments to check how the algorithms vary over the same NN.
    return rs


def _share_data(comm: MPI.Comm, send_data: List[float]) -> np.ndarray:
    """Share data to all processes"""
    # must use float64 for large int idxs. otherwise I get back the wrong numbers
    send_data = np.array(send_data * comm.size, dtype=np.float64)
    shared_data = np.empty(send_data.shape, dtype=np.float64)
    comm.Alltoall(send_data, shared_data)
    return shared_data
