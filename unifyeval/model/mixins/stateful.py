import abc
from collections import defaultdict, Iterable
from typing import List, Dict

from unifyeval.model.deep_model import DeepModelBase
from unifyeval.model.layers.layer_base import LayeredModel
from unifyeval.model.types import Tensor


class StatefulModel(DeepModelBase):
    """
    Class to represent  stateful models. The actual implementation of the hidden state is model-dependent.
    """

    def reset(self) -> "StatefulModel":
        """
        resets hidden states
        :return: model with hidden state reset to default state
        """
        self.set_state(self.get_default_state())
        return self

    @abc.abstractmethod
    def get_default_state(self) -> object:
        """
        get default (i.e. initial) state of current model
        """
        pass

    @abc.abstractmethod
    def set_state(self, state: object) -> "StatefulModel":
        """
        set state of given model and return model
        """
        pass

    @abc.abstractmethod
    def get_state(self) -> object:
        """
        get current state of model
        """
        pass

    def __enter__(self):
        """
        resets hidden states
        """
        return self.reset()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        resets hidden states
        """
        self.reset()


class StatefulLayeredModel(LayeredModel, StatefulModel):
    """
    Stateful model implementing the Layer API.
    """

    def _yield_stateful_layers(self):
        """
        yields all layers of the given model
        """
        for layer in self.layers:
            if isinstance(layer, StatefulModel):
                yield layer

    def get_default_state(self) -> List[object]:
        """
        get default state of the model as a (possibly nested) list of sub-states
        """
        return [layer.get_default_state() for layer in self._yield_stateful_layers()]

    def set_state(self, state: Iterable) -> "StatefulLayeredModel":
        """
        set state of model to a given value and return model.
        The given state has to be a (possibly nested) iterable over sub-states
        """
        for layer, next_state in zip(self._yield_stateful_layers(), state):
            layer.set_state(next_state)
        return self

    def get_state(self) -> List[object]:
        """
        get state of model as a (possibly nested) list over sub-states
        """
        return [layer.get_state() for layer in self._yield_stateful_layers()]

    @classmethod
    def from_components(cls, **kwargs) -> "StatefulLayeredModel":

        sorted_layers = StatefulLayeredModel._get_sorted_layers(**kwargs)
        return StatefulLayeredModel(layers=sorted_layers)

    def unroll(self, unrollable_kw: str, single_input_kw: str, **kwargs) -> Dict[str, List[Tensor]]:
        """
        given a keyword for unrollable data, that data is fed sequentially to the model,
        with the corresponding outputs collected in dictionary mapping from output keyword to a list of outputs
        :param unrollable_kw: keyword to extract unrollable data
        :param single_input_kw: keyword with which to push single data into model
        :param kwargs: additional kwargs
        :return: defaultdict mapping from output keyword to a list of outputs
        """
        # extract data
        batched_data = kwargs.pop(unrollable_kw)

        # send everything trough model
        aggregated_outputs = defaultdict(list)
        for batch in batched_data:
            for k, v in self.push(**{single_input_kw: batch}, **kwargs).items():
                aggregated_outputs[k].append(v)

        return aggregated_outputs
