from typing import Dict, Iterator

import numpy as np
import torch as t

from unifyeval.model.layers.layer_base import Layer
from unifyeval.model.types import Tensor


class LinearLayer(t.nn.Linear, Layer):
    """
    Layer that subclasses torch.nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int,
                 input_kw: str = "linear_input",
                 output_kw: str = "linear_output"):
        """
        :param in_features: number of input features
        :param out_features: number of output features
        :param input_kw: name of input data
        :param output_kw: name of output data
        """
        super().__init__(in_features, out_features)
        self.input_kw = input_kw
        self.output_kw = output_kw

    def push(self, **kwargs) -> dict:
        input = kwargs[self.input_kw]
        kwargs[self.output_kw] = self(input=input)
        return kwargs

    def get_components(self) -> dict:
        return {"input_kw": self.input_kw,
                "output_kw": self.output_kw,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "state_dict": self.state_dict()}

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy()) for name, p in self.named_parameters())

    @classmethod
    def from_components(cls, **kwargs) -> "Layer":
        state_dict = kwargs.pop("state_dict")
        layer = LinearLayer(**kwargs)
        layer.load_state_dict(state_dict)
        return layer

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return self.parameters()

    def train_mode(self) -> "LinearLayer":
        self.train()
        return self

    def eval_mode(self) -> "LinearLayer":
        self.eval()
        return self
