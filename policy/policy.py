from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from evo.noisetable import NoiseTable


class Policy(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, lr: float):
        super().__init__()
        # module.apply(init_normal) - this should belong to the NN

        self._module: torch.nn.Module = module
        self.lr = lr
        self.flat_params: np.ndarray = Policy.get_flat(module)
        self.n_params = len(self.flat_params)
        # self.controller_params = Policy.get_flat(module.controller)
        # self.m_n_params = len(Policy.get_flat(module.memory))
        # self.memory_params = Policy.get_flat(module.memory)
        # self.m_n_params = len(Policy.get_flat(module.memory))
        # self.mode = 0
        self.best = self.flat_params
        self.mu = np.mean(self.flat_params)
        self.std = np.std(self.flat_params)
        self.skew = stats.skew(self.flat_params)
        self.kurt = stats.kurtosis(self.flat_params)
        self.mode = 0

    def __len__(self):
        return len(self.flat_params)

    @staticmethod
    def get_flat(module: torch.nn.Module) -> np.ndarray:
        return torch.cat([t.flatten() for t in module.state_dict().values()]).numpy()

    def set_nn_params(self, params: np.ndarray) -> torch.nn.Module:
        with torch.no_grad():
            d = {}  # new state dict
            curr_params_idx = 0
            for name, weights in self._module.state_dict().items():
                n_params = torch.prod(torch.tensor(weights.shape))
                d[name] = torch.from_numpy(
                    np.reshape(params[curr_params_idx:curr_params_idx + n_params], weights.size()))
                curr_params_idx += n_params
            self._module.load_state_dict(d)
        return self._module

    def pheno(self, cfg, idxs: np.ndarray, noise_tbl: NoiseTable = None) -> np.ndarray:
        params = self.flat_params.copy()
        for n, i in enumerate(idxs):
            if i != 0:
                sgn = np.sign(i)
                noise = noise_tbl.get(abs(i)) if noise_tbl is not None else np.zeros(len(self))
                if cfg.noise.cholesky:
                    corr = np.array([[1, cfg.noise.corr],
                                     [cfg.noise.corr, 1]])
                    L = np.linalg.cholesky(corr).T  # np returns lower triangular and upper triangular is needed
                    X = np.stack((params, noise)).T
                    m = np.mean(X, axis=0)
                    s = np.std(X, axis=0)
                    X = (X - m) / s
                    Y = np.dot(X, L)
                    Y = Y * s + m
                    params = Y[:, 0]
                    noise = Y[:, 1]
                params += sgn * self.lr * noise  # adding two Gaussian is a Gaussian with increasing variance
        return params

    def forward(self, inp):
        self._module.forward(inp)

    def update_best(self, best):
        self.best = best
        self.mu = np.mean(self.best)
        self.std = np.std(self.best)
        self.skew = stats.skew(self.best)
        self.kurt = stats.kurtosis(self.best)
        count, bins = np.histogram(self.best, 1000, density=False)
        self.mode = bins[np.argmax(count)]
