from typing import List, Dict, Tuple, Iterator, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from unify_eval.model.deep_model import DeepModelBase
from unify_eval.model.mixins.classification import Classifier
from unify_eval.model.types import Tensor
from unify_eval.utils.label_mapper import LabelMapper


class Layer(DeepModelBase):

    def __init__(self, sub_layers: List[Tuple[str, "Layer"]] = None):
        if sub_layers is None:
            sub_layers = []
        self.sub_layers = sub_layers
        # set attribute names and set of names
        self.sub_layer_names = set()
        for name, layer in self.sub_layers:
            setattr(self, name, layer)
            self.sub_layer_names.add(name)

    def yield_sub_layers(self):
        for _, layer in self.sub_layers:
            yield layer

    def push(self, **kwargs) -> dict:
        for layer in self.yield_sub_layers():
            kwargs = layer.push(**kwargs)
        return kwargs

    def __add__(self, *other: Tuple[str, "Layer"]):
        return Layer(sub_layers=self.sub_layers + list(other))

    def get_components(self) -> dict:
        return dict((f"{i_layer}_{layer_name}", layer)
                    for i_layer, [layer_name, layer]
                    in enumerate(self.sub_layers))

    @classmethod
    def _get_sorted_layers(cls, **kwargs) -> List[Tuple[str, "Layer"]]:
        """
        internally used method to correctly re-organize the list of contained layers
        """
        # names are prefixed by "index_", so sort them by it
        indices = []
        names = []
        layers = []
        for name, layer in kwargs.items():
            # split index from actual name
            split = name.split("_")
            index = split[0]
            # join actual name back (might contain underscores)
            name = "_".join(split[1:])
            indices.append(int(index))
            names.append(name)
            layers.append(layer)

        # sort everything according to original indices
        sorted_indices = sorted(list(range(len(indices))), key=lambda x: indices[x])

        sorted_layers = [(names[index], layers[index]) for index in sorted_indices]
        return sorted_layers

    @classmethod
    def from_components(cls, **kwargs) -> "Layer":
        sub_layers = Layer._get_sorted_layers(**kwargs)
        return cls(sub_layers=sub_layers)

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        params = dict()
        for name, layer in self.sub_layers:
            for pname, p in layer.get_numpy_parameters().items():
                params[f"{name}/{pname}"] = p
        return params

    def get_optimizable_parameters(self) -> Iterator[Tensor]:

        for _, layer in self.sub_layers:
            for param in layer.get_optimizable_parameters():
                yield param

    def __getitem__(self, item) -> "Layer":
        assert type(item) == str or type(item) == int or type(item) == slice

        if isinstance(item, int):
            return self.sub_layers[item][1]
        if isinstance(item, str):
            assert item in vars(self), f"{item} not found as name for any sub-layer"
            return getattr(self, item)
        if isinstance(item, slice):
            return Layer(sub_layers=self.sub_layers[item])

    def __contains__(self, item: str):
        return item in self.sub_layer_names

    def _to_pretty_string(self, entire_string: str, offset: int) -> str:
        line_prefix = "\t" * offset
        for name, layer in self.sub_layers:
            entire_string += f"{line_prefix}{name}\n"
            entire_string = layer._to_pretty_string(entire_string=entire_string,
                                                    offset=offset + 1)
        return entire_string

    def __repr__(self):
        return self._to_pretty_string(entire_string="Layer\n", offset=1)

    def train_mode(self) -> "Layer":
        for layer in self.yield_sub_layers():
            layer.train_mode()
        return self

    def eval_mode(self) -> "Layer":
        for layer in self.yield_sub_layers():
            layer.eval_mode()
        return self

    def to_device(self, name: str) -> "Layer":
        self.current_device = name
        for layer in self.yield_sub_layers():
            layer.to_device(name)
        return self


class PytorchLayer(Layer, torch.nn.Module):
    def __init__(self,
                 input_kws: List[str],
                 output_kws: List[str]
                 ):
        Layer.__init__(self)
        torch.nn.Module.__init__(self)
        self.input_kws = input_kws
        self.output_kws = output_kws

    def push(self, **kwargs) -> dict:
        inputs = [kwargs[kw] for kw in self.input_kws]
        output = self(*inputs)
        if isinstance(output, tuple):
            for kw, out in zip(self.output_kws, *output):
                kwargs[kw] = out
        else:
            kwargs[self.output_kws[0]] = output
        return kwargs

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return self.parameters()

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((key, value.detach().cpu().numpy()) for key, value in self.named_parameters())

    def train_mode(self) -> "PytorchLayer":
        self.train()
        return self

    def eval_mode(self) -> "PytorchLayer":
        self.eval()
        return self

    def to_device(self, name: str) -> "PytorchLayer":
        super().to_device(name)
        self.to(name)
        return self


class PytorchWrapperLayer(Layer):
    def __init__(self,
                 pytorch_module: torch.nn.Module,
                 input_kws: List[str],
                 output_kws: List[str]):
        super().__init__()
        self.pytorch_module = pytorch_module
        self.input_kws = input_kws
        self.output_kws = output_kws

    def get_components(self) -> dict:
        return dict(pytorch_module=self.pytorch_module,
                    input_kws=self.input_kws,
                    output_kws=self.output_kws)

    @classmethod
    def from_components(cls, **kwargs) -> "PytorchWrapperLayer":
        return cls(**kwargs)

    def push(self, **kwargs) -> dict:
        inputs = [kwargs[kw] for kw in self.input_kws]
        output = self.pytorch_module(*inputs)
        if isinstance(output, tuple):
            for kw, out in zip(self.output_kws, *output):
                kwargs[kw] = out
        else:
            kwargs[self.output_kws[0]] = output
        return kwargs

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return self.pytorch_module.parameters()

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((key, value.detach().numpy()) for key, value in self.pytorch_module.named_parameters())

    def train_mode(self) -> "PytorchWrapperLayer":
        self.pytorch_module.train()
        return self

    def eval_mode(self) -> "PytorchWrapperLayer":
        self.pytorch_module.eval()
        return self

    def to_device(self, name: str) -> "PytorchWrapperLayer":
        super().to_device(name)
        self.pytorch_module.to(name)
        return self


class PTClassifierWrapperLayer(Classifier, PytorchWrapperLayer):
    def __init__(self, pipeline: Layer, label_mapper: LabelMapper):
        super().__init__(label_mapper)
        self.pipeline = pipeline
        self.xent = CrossEntropyLoss()
        self.opt = torch.optim.Adam(params=list(self.get_optimizable_parameters()))

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        return torch.nn.functional.softmax(self.get_logits(**kwargs).detach()).numpy()

    def get_logits(self, **kwargs) -> Tensor:
        return self.pipeline.push(**kwargs)["logits"]

    def train(self, **kwargs) -> "PTClassifierWrapperLayer":
        self.get_loss(**kwargs)["loss"].backward()
        self.opt.step()
        self.opt.zero_grad()
        return self

    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        logits = self.get_logits(**kwargs)
        target = torch.from_numpy(kwargs[kwargs["label_kw"]]).float()
        loss = self.xent(input=logits, target=target)
        return dict(loss=loss)
