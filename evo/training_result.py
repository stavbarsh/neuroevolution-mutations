from abc import ABC, abstractmethod
from typing import Sequence, List

import numpy as np


class TrainingResult(ABC):
    """Stores the results of a single training run"""

    def __init__(self, rewards: np.ndarray, positions: np.ndarray, obs: np.ndarray, done: bool):
        self.rewards: np.ndarray = rewards
        self.positions: np.ndarray = positions
        self.obs: np.ndarray = obs
        self.done: bool = done

    @abstractmethod
    def get_result(self) -> Sequence[float]:
        pass

    result: Sequence[float] = property(lambda self: self.get_result())
    reward = property(lambda self: sum(self.rewards))
    dist = property(lambda self: self.rewards[-1])  # this is not the distance
    behaviour = property(lambda self: self.positions[:, -1, :])  # the last position is a 2D XY vector


class RewardResult(TrainingResult):
    def get_result(self) -> List[float]:
        return [sum(self.rewards)]


class DistResult(TrainingResult):
    def get_result(self) -> List[float]:
        return [self.rewards[-1]]
