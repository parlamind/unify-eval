import abc
from typing import List, Tuple, Dict, Iterator, Set, Union

import numpy as np
import torch as t

from unifyeval.model.deep_model import DeepModelBase
from unifyeval.model.types import Tensor
from unifyeval.utils.other_utils import xor


class Layer(DeepModelBase):
    """
    Single Layer Base case class, to be subclassed by actual models.
    """

    @abc.abstractmethod
    def push(self, **kwargs) -> dict:
        """
        Pushes data through a layer. Actual definition of that data is model-specific
        """
        pass

    def get_components(self) -> dict:
        return dict()

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict()

    @classmethod
    def from_components(cls, **kwargs) -> "Layer":
        return Layer()

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return iter([])


class LayerContainer:
    """
    Class to manage a sequence of single Layers.
    """

    def __init__(self, named_layer_list: List[Tuple[str, Layer]]):
        self.named_layer_list: List[Tuple[str, Layer]] = []
        self.layer_names: Set[str] = set()
        for layer_name, layer in named_layer_list:
            setattr(self, layer_name, layer)
            self.named_layer_list.append((layer_name, layer))
            assert layer_name not in self.layer_names, f"No duplicate layer names allowed: {layer_name}"
            self.layer_names.add(layer_name)

    def __contains__(self, item) -> bool:
        """
        returns true if layer name is found among contained layers
        """
        return item in self.layer_names

    def __iter__(self) -> Iterator[Layer]:
        """
        returns iterator over contained layer objects
        """
        return iter(layer for _, layer in self.named_layer_list)

    def __getitem__(self, item):

        return self.layers(item)

    def named_layers(self, item):
        """
        returns
        """
        assert type(item) == str or type(item) == int or type(item) == slice

        if isinstance(item, int):
            return self.named_layer_list[item]
        if isinstance(item, str):
            assert item in vars(self)
            return (item, getattr(self, item))
        if isinstance(item, slice):
            return [x for x in self.named_layer_list[item]]

    def layers(self, item) -> Union[Layer, List[Layer]]:
        """
        Extracts layer objects given some name, index or slice.
        """
        assert type(item) == str or type(item) == int or type(item) == slice

        if isinstance(item, int):
            return self.named_layer_list[item][1]
        if isinstance(item, str):
            assert item in vars(self)
            return getattr(self, item)
        if isinstance(item, slice):
            return [x[1] for x in self.named_layer_list[item]]

    def extract(self, item) -> "LayerContainer":
        """
        Extracts layer objects given some name, index or slice and then returns them as a new LayerContained instance.
        """
        extracted_layers = self.named_layers(item)
        if not isinstance(extracted_layers, list):
            extracted_layers = [extracted_layers]
        return LayerContainer(extracted_layers)

    def __add__(self, other: "LayerContainer") -> "LayerContainer":
        """
        Concatenates two Layercontainer instances.
        """
        assert isinstance(other, LayerContainer)
        combined_layers = self.named_layer_list + other.named_layer_list
        return LayerContainer(named_layer_list=combined_layers)


class LayeredModel(Layer):
    """
    Model that contains a list of sub-models.
    """

    def __init__(self, layers: List[Tuple[str, Layer]] = None, layer_container: LayerContainer = None):
        """
        Creates a LayeredModel instance.
        Either a list of name - layer pairs or a predefined LayerContainer object should be passed
        :param layers: list of name - layer tuples. defaults to None
        :param layer_container: layer contained. defaults to None
        """
        assert xor(layers is None, layer_container is None)
        self.layers = LayerContainer(layers) if layers is not None else layer_container

    def push(self, **kwargs) -> dict:
        """
        pushes data consecutively through all contained layers.
        """
        x = kwargs
        for layer in self.layers:
            # print(f"x {x}")
            x = layer.push(**x)
        return x

    @classmethod
    def _get_sorted_layers(cls, **kwargs) -> List[Tuple[str, Layer]]:
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
    def from_components(cls, **kwargs) -> "LayeredModel":
        sorted_layers = LayeredModel._get_sorted_layers(**kwargs)
        return LayeredModel(layers=sorted_layers)

    def get_components(self) -> dict:
        return dict((f"{i_layer}_{layer_name}", layer)
                    for i_layer, [layer_name, layer]
                    in enumerate(self.layers.named_layer_list))

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((f"{layer_name}/{p_name}", p)
                    for layer_name, layer in self.layers.named_layer_list
                    for p_name, p in layer.get_numpy_parameters().items())

    def __getitem__(self, item):
        """
        Extracts layer objects given some name, index or slice and then returns them as a new LayeredModel instance.
        """
        return self.extract(item)

    def extract(self, item) -> "LayeredModel":
        """
        Extracts layer objects given some name, index or slice and then returns them as a new LayeredModel instance.
        """
        extracted_layer_container = self.layers.extract(item)
        return LayeredModel(layer_container=extracted_layer_container)

    def __add__(self, other: "LayeredModel") -> "LayeredModel":
        """
        Concatenates two Layered models.
        """
        assert isinstance(other, LayeredModel)
        return LayeredModel(layer_container=self.layers + other.layers)

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return (p for _, layer in self.layers.named_layer_list for p in layer.get_optimizable_parameters())

    def train_mode(self) -> "DeepModelBase":
        for layer in self.layers:
            layer.train_mode()
        return self

    def eval_mode(self) -> "DeepModelBase":
        for name, layer in self.layers.named_layer_list:
            layer.eval_mode()
        return self


class PytorchModuleLayer(Layer):
    """
    Wrapper around pytorch modules.
    """

    def __init__(self,
                 module: t.nn.Module,
                 input_names: List[str],
                 output_names: List[str]):
        """
        Constructs a PytorchModuleLayer from a pytorch module. Expects lists of input and output names
        """
        self.module = module
        self.input_names = input_names
        self.output_names = output_names

    def push(self, **kwargs) -> dict:
        outputs = self.module(*[kwargs[name] for name in self.input_names])
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        for name, output in zip(self.output_names, outputs):
            kwargs[name] = output
        return kwargs

    def get_components(self) -> dict:
        return dict(module=self.module,
                    input_names=self.input_names,
                    output_names=self.output_names)

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, parameter.detach().numpy()) for name, parameter in self.module.named_parameters())

    @classmethod
    def from_components(cls, **kwargs) -> "PytorchModuleLayer":
        return PytorchModuleLayer(**kwargs)

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        return self.module.parameters()

    def train_mode(self) -> "PytorchModuleLayer":
        self.module.train()
        return self

    def eval_mode(self) -> "PytorchModuleLayer":
        self.module.eval()
        return self
