import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
from typing import Optional

import numpy as np
import gym  # it makes a difference where if you import gym before torch!!!???
import torch

from mpi4py import MPI

from evo import ga
from evo.noisetable import NoiseTable
from policy.policy import Policy
from policy.nn import Actor, save_nn
from utils import utils
from evo.rankers import EliteRanker, MaxNormalizedRanker
from evo.reporters import LoggerReporter, ReporterSet, StdoutReporter
from evo.training_result import TrainingResult, RewardResult
from utils.utils import generate_seed, seed
from gym import Env

from utils.viz import graph
from policy.utils import plot_param_dist


def fit_fn(cfg, env, model: torch.nn.Module) -> TrainingResult:
    with torch.no_grad():
        mean = np.zeros(env.observation_space.shape)
        std = np.ones(env.observation_space.shape)
        if cfg.training.ob_norm:
            if np.all(env.observation_space.bounded_above * env.observation_space.bounded_below):
                std = np.max(np.abs([env.observation_space.high, env.observation_space.low]), axis=0)
            else:
                mean = model.obstat.mean
                std = model.obstat.std
        state, _ = env.reset(seed=cfg.env.seed)
        rews, obs, behv = [], [], []
        done = False
        truncated = False
        i = 0
        while not done and not truncated:
            state_t = torch.tensor((state - mean) / std, dtype=torch.float32).unsqueeze(dim=0)
            action_t = model.forward(state_t)
            action = action_t.numpy()[0]
            state, reward, done, truncated, info = env.step(action)
            x = info["x_position"] if "x_position" in info.keys() else 0.
            y = info["y_position"] if "y_position" in info.keys() else 0.
            behv.append((x, y))
            rews.append(reward)
            obs.append(state)
            i += 1
    return RewardResult(np.array(rews), np.array([behv]), np.array(obs), done)


def main():
    # Start MPI
    comm: MPI.Comm = MPI.COMM_WORLD

    # Load configuration file parser
    cfg_parser = utils.parse_args()
    if comm.size == 1:
        cfg_parser.config = 'configs/uber-ga.json'

    cfg = utils.load_config(cfg_parser.config)
    cfg.noise.seed = cfg_parser.force_seed if cfg_parser.force_seed is not None else cfg.noise.seed
    cfg.env.name = cfg_parser.force_env if cfg_parser.force_env is not None else cfg.env.name
    cfg.training.ob_norm = bool(
        cfg_parser.force_obnorm) if cfg_parser.force_obnorm is not None else cfg.training.ob_norm
    cfg.policy.lr = cfg_parser.force_lr if cfg_parser.force_lr is not None else cfg.policy.lr
    cfg.training.gens = cfg_parser.force_gens if cfg_parser.force_gens is not None else cfg.training.gens

    # The current working directory
    local_dir = os.path.dirname(__file__)

    # seeding
    cfg.noise.seed = generate_seed(comm) if cfg.noise.seed is None else cfg.noise.seed
    rs = seed(comm, cfg.noise.seed)

    # Make the experiment
    env = gym.make(cfg.env.name)
    out_dir = os.path.join(local_dir, 'saved', cfg_parser.config.split('/')[-1], cfg.env.name, str(cfg.noise.seed),
                           time.strftime("%j-%Y-%H:%M:%S"))
    if comm.rank == 0:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # init reporters
    reporter = ReporterSet(
        LoggerReporter(comm, cfg, out_dir),
        StdoutReporter(comm))
    # mlflow_reporter)

    reporter.print(cfg)
    reporter.print(out_dir)
    reporter.print(f'seed:{cfg.noise.seed}')

    # init variables
    in_size, out_size = env.observation_space.shape[0], env.action_space.shape[0]
    best_gnrl = 0
    best_reward = -np.inf
    parent_ind = np.zeros((1, 1), dtype=np.int32)
    parent_res = []
    policy = None

    # create/load n_meta_polices and scatter to cores
    if comm.rank == 0:
        policy = Policy(Actor(cfg.policy.hidden_ctrl, int(in_size), int(out_size)), cfg.policy.lr)
        parent_res = []
        res = fit_fn(cfg, env, policy.set_nn_params(policy.best))
        parent_res.append([res.result + [0]])
        best_reward = max(res.reward, best_reward)

    policy = comm.bcast(policy)
    best_reward = comm.bcast(best_reward)
    parent_res = comm.bcast(parent_res)

    # initialize and share the NT
    if comm.size == 1:
        nt: NoiseTable = NoiseTable(n_params=policy.n_params,
                                    noise=NoiseTable.make_noise(int(cfg.noise.tbl_size), cfg.noise.mu,
                                                                cfg.noise.std, 0))
    else:
        nt: NoiseTable = NoiseTable.create_shared(comm, int(cfg.noise.tbl_size), cfg.noise.mu,
                                                  cfg.noise.std, policy.n_params, reporter,
                                                  0)

    # Start experiment
    for gen in range(cfg.training.gens):
        if comm.rank == 0:
            t = time.time()

        ranker = EliteRanker(MaxNormalizedRanker(), cfg.training.elite)
        reporter.start_gen()
        tr, elite_inds, elite_res = ga.step(cfg, comm, env, parent_res, parent_ind, policy, nt,
                                                                   fit_fn, rs, ranker, reporter)

        reward = tr.reward
        parent_ind = elite_inds
        parent_res = elite_res

        update_rule = reward > best_reward

        if update_rule:
            best_reward = reward

        if comm.rank == 0:
            reporter.print(f'mu: {policy.mu}')
            reporter.print(f'std: {policy.std}')
            reporter.print(f'skew: {policy.skew}')
            reporter.print(f'kurt: {policy.kurt}')
            reporter.print(f'mode: {policy.mode}')
            reporter.print(f'count: {policy._module.obstat.count:0.0f}')
            if update_rule:
                reporter.print(f'saving policy with:{reward:0.2f}')
                save_nn(policy.set_nn_params(policy.best), out_dir)

        reporter.end_gen()

        if comm.rank == 0:
            print(f'total time: {time.time() - t}')

    # mlflow.end_run()  # ending the outer mlfow run
    if comm.rank == 0:
        graph(f'{out_dir}/{reporter.reporters[0].log_name}.log',
              view=False, filename='plot_graph', filepath=out_dir, fig_n=1)
        plot_param_dist(policy.best, filename='params_dist', filepath=out_dir, fig_n=0)


if __name__ == '__main__':
    # no variable is global
    main()
    print('Finished!')
