from __future__ import annotations

import time
from collections.abc import Callable
from typing import Tuple

import numpy as np
from mpi4py import MPI
from numpy import ndarray
from numpy.random import RandomState

from evo.noisetable import NoiseTable
from policy.policy import Policy
from evo.rankers import Ranker, EliteRanker
from evo.reporters import StdoutReporter, Reporter
from evo.training_result import TrainingResult
from gym import Env

from utils.utils import _share_data


def step(cfg,
         comm: MPI.Comm,
         env: Env,
         parent_res: ndarray,
         parent_ind: ndarray,
         policy: Policy,
         nt: NoiseTable,
         fit_fn: Callable,
         rs: RandomState = np.random.RandomState(),
         ranker: Ranker = EliteRanker,
         reporter: Reporter = StdoutReporter(MPI.COMM_WORLD)) -> [TrainingResult]:
    assert cfg.training.policies_per_gen % comm.size == 0
    eps_per_proc = int((cfg.training.policies_per_gen / comm.size))

    gen_start = time.time()
    fits, inds, ob_inc = test_params(cfg, comm, eps_per_proc, env, parent_res, parent_ind, policy, nt, fit_fn, rs)
    # all processes comes out with the results so the rest is also the same (i.e. rank, elite etc.)
    elite_fits, elite_inds = rank(ranker, fits, inds)
    best_child = calc_best_child(cfg, comm, env, elite_inds, nt, fit_fn, policy)
    # calc obstat after best child is found
    s, ssq, c = ob_inc
    policy._module.obstat.inc(s, ssq, c)
    reporter.log_gen(fits, best_child, time.time() - gen_start)
    return best_child, elite_inds, elite_fits


def test_params(cfg, comm: MPI.Comm, n: int, env: Env, parent_res: ndarray, parent_ind: ndarray, policy: Policy,
                nt: NoiseTable, fit_fn: Callable, rs: RandomState) -> Tuple[ndarray, ndarray, list]:
    res = []
    inds = []
    # perform N mutations to random parents
    for _ in range(n):
        c_idxs = parent_ind[rs.randint(0, len(parent_ind))].tolist()  # parent_ind is immutable!
        sgn = rs.choice([-1, 1]) if cfg.noise.swap else 1
        c_idxs.append(sgn * nt.sample_idx(rs))
        params = policy.pheno(cfg, c_idxs, nt)
        # for each noise ind sampled, both add and subtract the noise
        res.append(fit_fn(cfg, env, policy.set_nn_params(params)))
        inds.append(c_idxs)

    # averaged generality results - 1 objectives
    send_results = [r.result + i for r, i in zip(res, inds)]

    results = _share_data(comm, send_results)

    all_fits = np.expand_dims(results[:, 0], axis=-1)
    all_inds = results[:, 1:].astype(np.int32)

    # normalize states
    all_obs = [r.obs for r in res]
    all_obs = np.vstack(all_obs)
    s = all_obs.sum(axis=0)
    ssq = np.square(all_obs).sum(axis=0)
    c = all_obs.shape[0]
    s = _share_data(comm, [s.tolist()]).sum(axis=0)
    ssq = _share_data(comm, [ssq.tolist()]).sum(axis=0)
    c = _share_data(comm, [c]).sum(axis=0)
    ob_inc = [s, ssq, c]

    if cfg.training.with_parents:
        all_fits = np.concatenate((all_fits, parent_res))
        all_inds = np.concatenate((all_inds, [np.append(p, [0]) for p in parent_ind]))

    return all_fits, all_inds, ob_inc


def calc_best_child(cfg, comm, env, inds, nt, fit_fn, policy) -> TrainingResult:
    """Change the policy according to the best results (depands on the results)"""
    best_inds = inds[-1]
    policy.update_best(policy.pheno(cfg, best_inds, nt))
    best_child = comm.bcast(fit_fn(cfg, env, policy.set_nn_params(policy.best)))
    return best_child


def rank(ranker, fits, inds):
    ranker.rank(fits, inds)
    elite_fits = ranker.fits[ranker.elite_fit_inds]
    elite_inds = ranker.noise_inds.astype(np.int32)
    return elite_fits, elite_inds
