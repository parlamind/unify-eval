from abc import ABC
from typing import List, Union

import numpy as np
import torch as t

from unifyeval.model.deep_model import DeepModel


class DiscriminateLearningRateModel(DeepModel, ABC):

    def __init__(self,
                 all_parameter_groups: List[List[t.nn.Parameter]],
                 base_lr: float):

        self.base_lr = base_lr

        self.all_parameter_groups = all_parameter_groups
        self.group_is_active = [True for _ in all_parameter_groups]
        self.n_groups = len(self.group_is_active)

        self._setup_optimizer()
        self.optimizer.zero_grad()

    def _setup_optimizer(self):

        self.optimizer = t.optim.SGD([
            {"params": parameters, "lr": self.base_lr} for parameters, is_active in
            zip(self.all_parameter_groups, self.group_is_active)
            if is_active], lr=self.base_lr)
        self.optimizer.zero_grad()

    def freeze(self, indices: Union[slice, int] = None):
        indices = indices if indices is not None else np.arange(self.n_groups)

        if isinstance(indices, int):
            self.group_is_active[indices] = False
        else:
            self.group_is_active[np.arange(len(self.group_is_active))[indices]] = False
        self._setup_optimizer()

    def unfreeze(self, indices: slice = None):
        indices = indices if indices is not None else np.arange(self.n_groups)
        if isinstance(indices, int):
            self.group_is_active[indices] = True
        else:
            self.group_is_active[np.arange(len(self.group_is_active))[indices]] = True
        self._setup_optimizer()

    def set_learning_rate(self, index: int = None, learning_rate: float = 1e-4) -> "DiscriminateLearningRateModel":
        if index is not None:
            self.optimizer.param_groups[index]["lr"] = learning_rate
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate
        return self

    def set_all_learning_rates(self, learning_rates: List[float]) -> "DiscriminateLearningRateModel":
        assert len(learning_rates) == len(self.optimizer.param_groups)
        for i, lr in enumerate(learning_rates):
            self.set_learning_rate(index=i, learning_rate=lr)
        return self
