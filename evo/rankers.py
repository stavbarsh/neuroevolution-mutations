from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def rank(x: np.ndarray):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].

    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


class Ranker(ABC):
    """Ranks all fitnesses obtained in a generation"""

    def __init__(self):
        self.fits: Optional[np.ndarray] = None
        self.noise_inds: Optional[np.ndarray] = None
        self.ranked_fits: Optional[np.ndarray] = None
        self.n_fits_ranked: int = 0

    @abstractmethod
    def _rank(self, x: np.ndarray) -> np.ndarray:
        """Ranks self.fits"""
        pass

    def _pre_rank(self, fits: np.ndarray, noise_inds: np.ndarray):
        self.fits = fits
        self.noise_inds = noise_inds

    def _post_rank(self, ranked_fits: np.ndarray) -> np.ndarray:
        self.n_fits_ranked = ranked_fits.size

    def rank(self, fits: np.ndarray, noise_inds: np.ndarray) -> np.ndarray:
        self._pre_rank(fits, noise_inds)
        self.ranked_fits = self._rank(self.fits)
        self._post_rank(self.ranked_fits)
        return self.ranked_fits


class MaxNormalizedRanker(Ranker):
    def _rank(self, x: np.ndarray) -> np.ndarray:
        y = rank(x.ravel()).reshape(x.shape).astype(np.float32)
        return np.squeeze(y / np.max(y))


class EliteRanker(Ranker):
    def __init__(self, ranker: Ranker, elite_truncation: int):
        super().__init__()
        # assert 0 <= elite_percent <= 1
        assert elite_truncation > 0
        self.ranker = ranker
        self.elite_truncation = elite_truncation
        self.elite_fit_inds = None

    def _rank(self, x: np.ndarray) -> np.ndarray:
        ranked = self.ranker._rank(x)
        # n_elite = max(1, int(ranked.size * self.elite_percent))
        n_elite = self.elite_truncation
        # self.elite_fit_inds = np.argpartition(ranked, -n_elite)[-n_elite:]
        self.elite_fit_inds = np.argsort(ranked)[-n_elite:]
        # setting the noise inds to only be the inds of the elite
        return ranked[self.elite_fit_inds]

    def _post_rank(self, ranked_fits: np.ndarray) -> np.ndarray:
        self.n_fits_ranked = ranked_fits.size
        self.noise_inds = self.noise_inds[self.elite_fit_inds % len(self.noise_inds)]
        return ranked_fits
